import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.models.schemas import DocumentContent, DocumentInfo, IngestionResult

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/documents/upload", response_model=list[IngestionResult])
def upload(request: Request, files: list[UploadFile] = File(...)):
    services = request.app.state.services
    results = []
    for file in files:
        try:
            # Read at most limit+1, rather than loading arbitrarily large uploads into RAM.
            content = file.file.read(services.settings.max_upload_size_mb * 1024 * 1024 + 1)
            with services.ingestion.lock:
                results.append(services.ingestion.ingest(file.filename or "", content))
        except ValueError as exc:
            results.append(dict(filename=file.filename or "", status="error", error=str(exc)))
        except Exception as exc:
            log.error("Ingestion failed (%s)", type(exc).__name__)
            results.append(
                dict(
                    filename=file.filename or "",
                    status="error",
                    error="Indexing failed. Check model availability, disk space and backend logs; retry the file.",
                )
            )
        finally:
            file.file.close()
    return results


@router.get("/documents", response_model=list[DocumentInfo])
def documents(request: Request):
    services = request.app.state.services
    with services.ingestion.lock:
        names = services.ingestion.source_names()
        return [
            dict(record, source_name=names.get(record["document_id"]))
            for record in services.ingestion.manifest.all()
        ]


@router.delete("/documents/{document_id}")
def delete(document_id: str, request: Request):
    try:
        request.app.state.services.ingestion.delete(document_id)
    except KeyError:
        raise HTTPException(404, "Document not found") from None
    return {"deleted": document_id}


@router.put("/documents/{document_id}", response_model=IngestionResult)
def replace(document_id: str, request: Request, file: UploadFile = File(...)):
    """Replace an active document only after the new version is fully indexed."""
    services = request.app.state.services
    try:
        content = file.file.read(services.settings.max_upload_size_mb * 1024 * 1024 + 1)
        return services.ingestion.ingest(file.filename or "", content, document_id)
    except KeyError:
        raise HTTPException(404, "Active document not found") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    finally:
        file.file.close()


@router.get("/documents/{document_id}/versions", response_model=list[DocumentInfo])
def versions(document_id: str, request: Request):
    try:
        return request.app.state.services.ingestion.versions(document_id)
    except KeyError:
        raise HTTPException(404, "Document not found") from None


@router.get("/documents/{document_id}/content", response_model=DocumentContent)
def content(document_id: str, request: Request):
    try:
        return request.app.state.services.ingestion.content(document_id)
    except (KeyError, FileNotFoundError):
        raise HTTPException(404, "Imported document content not found") from None


@router.get("/sources/notes/audit")
def source_audit(request: Request):
    services = request.app.state.services
    if services.settings.data_profile != "local":
        raise HTTPException(403, "Saved-note audit is available only in the private local profile")
    from backend.ingestion.source_audit import audit_notes

    with services.ingestion.lock:
        return audit_notes(services.ingestion)


@router.post("/index-jobs", status_code=202)
def submit_index_job(request: Request, file: UploadFile = File(...)):
    services = request.app.state.services
    try:
        content = file.file.read(services.settings.max_upload_size_mb * 1024 * 1024 + 1)
        return services.jobs.submit(file.filename or "", content)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    finally:
        file.file.close()


@router.get("/index-jobs")
def list_index_jobs(request: Request):
    return request.app.state.services.jobs.list()


@router.post("/index-jobs/{identifier}/retry")
def retry_index_job(identifier: str, request: Request):
    try:
        return request.app.state.services.jobs.retry(identifier)
    except KeyError:
        raise HTTPException(404, "Index job not found") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
