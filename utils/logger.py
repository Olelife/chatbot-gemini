import logging
import json
import uuid
from datetime import datetime

logger = logging.getLogger("ole-assistant")
logger.setLevel(logging.INFO)

class CloudLogFormatter(logging.Formatter):
    def format(self, record):
        trace_id = getattr(record, "trace_id", None)

        log_data = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": trace_id,
            "module": record.module,
        }

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(CloudLogFormatter())
logger.addHandler(handler)

def log_info(message, **kwargs):
    extra = {"extra_data": kwargs}
    logger.info(message, extra=extra)

def log_error(message, **kwargs):
    extra = {"extra_data": kwargs}
    logger.error(message, extra=extra)

def generate_trace_id():
    return str(uuid.uuid4())