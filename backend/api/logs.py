"""API endpoint for fetching Logfire logs."""

import httpx
from fastapi import HTTPException
import logfire

from backend.config import settings


LOGFIRE_API_BASE = "https://logfire-us.pydantic.dev"  # Use logfire-eu.pydantic.dev for EU region


async def fetch_logfire_logs(trace_id: str) -> dict:
    """
    Fetch logs from Logfire API for a specific trace ID.
    
    Args:
        trace_id: The OpenTelemetry trace ID (32-char hex string)
        
    Returns:
        Dictionary containing the logs data from Logfire
        
    Raises:
        HTTPException: If the API request fails or auth is missing
    """
    if not settings.logfire_read_token:
        raise HTTPException(
            status_code=501,
            detail="Logfire read token not configured. Set LOGFIRE_READ_TOKEN environment variable."
        )
    
    # SQL query to fetch all records for this trace
    # Logfire stores data in the 'records' table
    # See: https://logfire.pydantic.dev/docs/how-to-guides/query-api/
    query = f"""
        SELECT 
            start_timestamp,
            message,
            level,
            span_name,
            span_id,
            parent_span_id,
            attributes,
            service_name,
            trace_id
        FROM records
        WHERE trace_id = '{trace_id}'
        ORDER BY start_timestamp ASC
        LIMIT 1000
    """
    
    headers = {
        "Authorization": f"Bearer {settings.logfire_read_token}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{LOGFIRE_API_BASE}/v1/query",
                params={"sql": query},
                headers=headers
            )
            
            if response.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid Logfire read token")
            elif response.status_code == 403:
                raise HTTPException(status_code=403, detail="Insufficient permissions for Logfire API")
            elif response.status_code != 200:
                logfire.error(f"Logfire API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Logfire API error: {response.text}"
                )
            
            data = response.json()
            
            # Logfire returns data in columnar format, need to convert to rows
            columns = data.get("columns", [])
            
            # Transform columnar format to row format
            rows = []
            if columns:
                # Get the number of rows from the first column's values
                num_rows = len(columns[0].get("values", []))
                
                # Transpose columnar data to row format
                for i in range(num_rows):
                    row = {}
                    for col in columns:
                        col_name = col.get("name")
                        values = col.get("values", [])
                        row[col_name] = values[i] if i < len(values) else None
                    rows.append(row)
            
            logfire.info(f"Fetched {len(rows)} log records for trace {trace_id}")
            
            return {
                "trace_id": trace_id,
                "logs": rows,
                "record_count": len(rows)
            }
            
    except httpx.TimeoutException:
        logfire.error(f"Timeout fetching logs for trace {trace_id}")
        raise HTTPException(status_code=504, detail="Logfire API timeout")
    except httpx.RequestError as e:
        logfire.error(f"Error fetching logs for trace {trace_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to connect to Logfire API: {str(e)}")
