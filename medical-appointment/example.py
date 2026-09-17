"""Local, deadline-bounded medical consultation inference."""

import logging

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from pipeline.core import floor_response
from pipeline.runtime import Pipeline
from utils import validate_response

logger = logging.getLogger(__name__)
_pipeline = Pipeline()


def start():
    _pipeline.start()


def stop():
    _pipeline.close()


def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    try:
        response = _pipeline.predict(request)
        validate_response(response, expected_count=len(request.questions))
        return response
    except Exception:
        logger.exception('Returning a valid guess after prediction failure')
        return floor_response(len(request.questions))
