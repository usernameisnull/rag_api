# app/routes/document_routes.py
import os
import hashlib
import traceback
import aiofiles
import aiofiles.os
from shutil import copyfileobj
from typing import List, Iterable, Dict, Any
from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
)
from langchain_core.documents import Document
from langchain_core.runnables import run_in_executor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache
import asyncio
import time

from app.config import (
    logger,
    vector_store,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTRACT_IMAGES,
    CONCURRENT_BATCH_SIZE,
    SUCCESS_RATE_THRESHOLD,
    REQUEST_TIMEOUT,
)
from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    DocumentResponse,
    QueryMultipleBody,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import get_loader, clean_text, process_documents
from app.utils.health import is_health_ok

router = APIRouter()


@router.get("/ids")
async def get_all_ids():
    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids()
        else:
            ids = vector_store.get_all_ids()

        return list(set(ids))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_all_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to get all IDs | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            return {"status": "UP"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str] = Query(...)):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(ids)
            documents = await vector_store.get_documents_by_ids(ids)
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure all requested ids exist
        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s",
            ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(document_ids: List[str] = Body(...)):
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(document_ids)
            await vector_store.delete(ids=document_ids)
        else:
            existing_ids = vector_store.get_filtered_ids(document_ids)
            vector_store.delete(ids=document_ids)

        if not all(id in existing_ids for id in document_ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        file_count = len(document_ids)
        return {
            "message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in delete_documents | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to delete documents | IDs: %s | Error: %s | Traceback: %s",
            document_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


# Cache the embedding function with LRU cache
@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = (
            body.entity_id if body.entity_id else request.state.user.get("id")
        )

    authorized_documents = []

    try:
        embedding = get_cached_query_embedding(body.query)

        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter={"file_id": body.file_id},
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter={"file_id": body.file_id}
            )

        if not documents:
            return authorized_documents

        document, score = documents[0]
        doc_metadata = document.metadata
        doc_user_id = doc_metadata.get("user_id")

        if doc_user_id is None or doc_user_id == user_authorized:
            authorized_documents = documents
        else:
            # If using entity_id and access denied, try again with user's actual ID
            if body.entity_id and hasattr(request.state, "user"):
                user_authorized = request.state.user.get("id")
                if doc_user_id == user_authorized:
                    authorized_documents = documents
                else:
                    if body.entity_id == doc_user_id:
                        logger.warning(
                            f"Entity ID {body.entity_id} matches document user_id but user {user_authorized} is not authorized"
                        )
                    else:
                        logger.warning(
                            f"Access denied for both entity ID {body.entity_id} and user {user_authorized} to document with user_id {doc_user_id}"
                        )
            else:
                logger.warning(
                    f"Unauthorized access attempt by user {user_authorized} to a document with user_id {doc_user_id}"
                )

        return authorized_documents

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


def generate_digest(page_content: str):
    hash_obj = hashlib.md5(page_content.encode())
    return hash_obj.hexdigest()


async def process_batch(
    vector_store: Any,
    batch_docs: List[Document],
    batch_file_ids: List[str],
    file_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """处理单个批次的文档"""
    try:
        if isinstance(vector_store, AsyncPgVector):
            # 使用配置的超时时间
            try:
                ids = await asyncio.wait_for(
                    vector_store.aadd_documents(batch_docs, ids=batch_file_ids),
                    timeout=REQUEST_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Request timeout while storing data in vector DB | File ID: %s | User ID: %s | Timeout: %d seconds",
                    file_id,
                    user_id,
                    REQUEST_TIMEOUT
                )
                return {"success": False, "error": f"Request timeout after {REQUEST_TIMEOUT} seconds"}
        else:
            ids = vector_store.add_documents(batch_docs, ids=batch_file_ids)
        return {"success": True, "ids": ids}
    except Exception as e:
        logger.error(
            "Failed to store data in vector DB for a batch | File ID: %s | User ID: %s | Error: %s | Traceback: %s",
            file_id,
            user_id,
            str(e),
            traceback.format_exc(),
        )
        return {"success": False, "error": str(e)}


async def add_documents_to_vector_store(
    docs: List[Document],
    file_id: str,
    user_id: str,
    vector_store: Any,
) -> Dict[str, Any]:
    """将文档添加到向量存储中，支持并发处理和成功率检查"""
    start_time = time.time()
    # 首先对文档进行分块处理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(docs)

    # 准备文档元数据
    processed_docs = [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "digest": generate_digest(doc.page_content),
                **(doc.metadata or {}),
            },
        )
        for doc in documents
    ]

    # 将文档分成批次
    batch_size = 10
    all_results = []
    all_ids = []
    
    # 创建任务队列
    task_queue = asyncio.Queue()
    for i in range(0, len(processed_docs), batch_size):
        batch_docs = processed_docs[i : i + batch_size]
        batch_file_ids = [file_id] * len(batch_docs)
        await task_queue.put((batch_docs, batch_file_ids))
    
    total_tasks = task_queue.qsize()
    completed_tasks = 0
    
    # 创建信号量来控制并发
    semaphore = asyncio.Semaphore(CONCURRENT_BATCH_SIZE)
    
    async def worker():
        nonlocal completed_tasks
        while True:
            try:
                batch_docs, batch_file_ids = await task_queue.get()
                async with semaphore:
                    result = await process_batch(vector_store, batch_docs, batch_file_ids, file_id, user_id)
                    all_results.append(result)
                    if result["success"]:
                        all_ids.extend(result["ids"])
                    
                    completed_tasks += 1
                    logger.info(
                        f"文档处理进度 | 文件ID: {file_id} | 已完成: {completed_tasks}/{total_tasks} | "
                        f"成功率: {(len([r for r in all_results if r['success']]) / completed_tasks * 100):.2f}%"
                    )
            except asyncio.CancelledError:
                break
            finally:
                task_queue.task_done()
    
    # 创建并启动工作协程
    workers = [asyncio.create_task(worker()) for _ in range(CONCURRENT_BATCH_SIZE)]
    
    # 等待所有任务完成
    await task_queue.join()
    
    # 取消所有工作协程
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    
    # 统计成功和失败的数量
    successful_results = [r for r in all_results if r["success"]]
    failed_results = [r for r in all_results if not r["success"]]
    
    # 计算成功率
    success_rate = (len(successful_results) / total_tasks) * 100 if total_tasks > 0 else 0
    
    # 最终处理结果日志
    total_time = time.time() - start_time
    logger.info(
        f"文档处理完成 | 文件ID: {file_id} | 总批次数: {total_tasks} | "
        f"成功: {len(successful_results)} | 失败: {len(failed_results)} | 成功率: {success_rate:.2f}% | "
        f"总耗时: {total_time:.2f}秒"
    )
    
    # 检查是否达到成功率阈值
    if success_rate >= SUCCESS_RATE_THRESHOLD:
        return {
            "message": "Documents added successfully",
            "ids": all_ids,
            "success_rate": success_rate,
            "failed_batches": len(failed_results),
            "total_batches": total_tasks
        }
    else:
        error_messages = [r["error"] for r in failed_results]
        return {
            "message": "Failed to add documents due to low success rate",
            "error": f"Success rate {success_rate}% is below threshold {SUCCESS_RATE_THRESHOLD}%",
            "failed_batches": len(failed_results),
            "total_batches": total_tasks,
            "error_details": error_messages
        }


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, entity_id: str = None
):
    # Check if the file exists
    if not os.path.exists(document.filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    try:
        loader, known_type = get_loader(
            document.filename, document.file_content_type, document.filepath
        )
        data = loader.load()
        result = await add_documents_to_vector_store(data, document.file_id, user_id, vector_store)

        if result:
            return {
                "status": True,
                "file_id": document.file_id,
                "filename": document.filename,
                "known_type": known_type,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=ERROR_MESSAGES.DEFAULT(),
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_local_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )


@router.post("/embed")
async def embed_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    response_status = True
    response_message = "File processed successfully."
    known_type = None
    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    temp_base_path = os.path.join(RAG_UPLOAD_DIR, user_id)
    os.makedirs(temp_base_path, exist_ok=True)
    temp_file_path = os.path.join(RAG_UPLOAD_DIR, user_id, file.filename)

    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024  # 64 KB
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )

    try:
        loader, known_type, file_ext = get_loader(
            file.filename, file.content_type, temp_file_path
        )
        data = loader.load()
        result = await add_documents_to_vector_store(data, file_id, user_id, vector_store)

        if not result:
            response_status = False
            response_message = "Failed to process/store the file data."
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
        elif "error" in result:
            response_status = False
            response_message = "Failed to process/store the file data."
            if isinstance(result["error"], str):
                response_message = result["error"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An unspecified error occurred.",
                )
    except HTTPException as http_exc:
        response_status = False
        response_message = f"HTTP Exception: {http_exc.detail}"
        logger.error(
            "HTTP Exception in embed_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        response_status = False
        response_message = f"Error during file processing: {str(e)}"
        logger.error(
            "Error during file processing: %s\nTraceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        try:
            await aiofiles.os.remove(temp_file_path)
        except Exception as e:
            logger.error(
                "Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s",
                temp_file_path,
                str(e),
                traceback.format_exc(),
            )

    return {
        "status": response_status,
        "message": response_message,
        "file_id": file_id,
        "filename": file.filename,
        "known_type": known_type,
    }


@router.get("/documents/{id}/context")
async def load_document_context(id: str):
    ids = [id]
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(ids)
            documents = await vector_store.get_documents_by_ids(ids)
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        # Ensure the requested id exists
        if not all(id in existing_ids for id in ids):
            raise HTTPException(
                status_code=404, detail="The specified file_id was not found"
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No document found for the given ID"
            )

        return process_documents(documents)
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in load_document_context | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error loading document context | Document ID: %s | Error: %s | Traceback: %s",
            id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    file_id: str = Form(...),
    uploaded_file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    temp_file_path = os.path.join(RAG_UPLOAD_DIR, uploaded_file.filename)

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    try:
        with open(temp_file_path, "wb") as temp_file:
            copyfileobj(uploaded_file.file, temp_file)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )

    try:
        loader, known_type = get_loader(
            uploaded_file.filename, uploaded_file.content_type, temp_file_path
        )

        data = loader.load()
        result = await add_documents_to_vector_store(data, file_id, user_id, vector_store)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process/store the file data.",
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_file_upload | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during file processing | File: %s | Error: %s | Traceback: %s",
            uploaded_file.filename,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )
    finally:
        os.remove(temp_file_path)

    return {
        "status": True,
        "message": "File processed successfully.",
        "file_id": file_id,
        "filename": uploaded_file.filename,
        "known_type": known_type,
    }


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(body: QueryMultipleBody):
    try:
        # Get the embedding of the query text
        embedding = get_cached_query_embedding(body.query)

        # Perform similarity search with the query embedding and filter by the file_ids in metadata
        if isinstance(vector_store, AsyncPgVector):
            documents = await run_in_executor(
                None,
                vector_store.similarity_search_with_score_by_vector,
                embedding,
                k=body.k,
                filter={"file_id": {"$in": body.file_ids}},
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter={"file_id": {"$in": body.file_ids}}
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given query"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))
