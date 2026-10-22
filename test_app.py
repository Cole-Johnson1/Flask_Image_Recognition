"""Pytest suite for the Flask routes and the image preprocessing/prediction helpers."""
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from model import preprocess_img, predict_result


def make_image_bytes(mode="RGB", size=(50, 50), fmt="PNG"):
    """Build an in-memory image file of the given colour mode."""
    buffer = BytesIO()
    Image.new(mode, size).save(buffer, fmt)
    buffer.seek(0)
    return buffer


def post_file(client, file_obj, name="digit.png"):
    """POST a file to the prediction route as multipart form data."""
    return client.post(
        "/prediction",
        data={"file": (file_obj, name)},
        content_type="multipart/form-data",
    )


# ---- Flask routes: normal cases ----
def test_home_page_loads(client):
    """The home page is served and contains the upload form."""
    response = client.get("/")
    assert response.status_code == 200
    assert b'type="file"' in response.data


def test_valid_upload_returns_prediction(client):
    """A valid image yields the predicted digit (model mocked)."""
    with patch("app.predict_result", return_value=7) as mocked:
        response = post_file(client, make_image_bytes())
    assert response.status_code == 200
    assert b">7<" in response.data
    mocked.assert_called_once()


def test_real_sample_image_is_predicted_end_to_end(client):
    """A real sample image goes through the actual model and returns a digit."""
    with open("test_images/2/Sign 2 (97).jpeg", "rb") as handle:
        response = post_file(client, handle, "sign.jpeg")
    assert response.status_code == 200
    assert b"File cannot be processed" not in response.data


# ---- Flask routes: edge and error cases ----
def test_missing_file_is_handled(client):
    """A request without the 'file' field shows the error message."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_non_image_file_is_handled(client):
    """A text file is rejected with the error message."""
    response = post_file(client, BytesIO(b"not an image"), "invalid.txt")
    assert b"File cannot be processed." in response.data


def test_empty_file_is_handled(client):
    """A zero-byte upload is rejected with the error message."""
    response = post_file(client, BytesIO(b""), "empty.png")
    assert b"File cannot be processed." in response.data


def test_get_on_prediction_route_not_allowed(client):
    """The prediction route only accepts POST."""
    assert client.get("/prediction").status_code == 405


def test_unknown_route_returns_404(client):
    """Unknown URLs return 404."""
    assert client.get("/does-not-exist").status_code == 404


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_upload_with_non_rgb_image_is_processed(client, mode):
    """Transparent, grayscale and palette images must still be predicted."""
    with patch("app.predict_result", return_value=3):
        response = post_file(client, make_image_bytes(mode))
    assert b">3<" in response.data


# ---- Preprocessing and prediction ----
def test_preprocess_img_shape_and_range():
    """Preprocessing returns a (1, 224, 224, 3) array scaled to [0, 1]."""
    result = preprocess_img(make_image_bytes("RGB", (300, 100)))
    assert result.shape == (1, 224, 224, 3)
    assert result.min() >= 0.0 and result.max() <= 1.0


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_preprocess_img_handles_other_colour_modes(mode):
    """Preprocessing converts any colour mode to three channels."""
    assert preprocess_img(make_image_bytes(mode)).shape == (1, 224, 224, 3)


def test_preprocess_img_rejects_invalid_data():
    """Non-image data raises an error instead of silently succeeding."""
    with pytest.raises(Exception):  # pylint: disable=broad-except
        preprocess_img(BytesIO(b"garbage"))


def test_predict_result_returns_argmax_of_model_output():
    """predict_result returns the index of the highest class probability."""
    fake_output = np.array([[0.05, 0.1, 0.05, 0.7, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]])
    with patch("model.model.predict", return_value=fake_output):
        assert predict_result(np.zeros((1, 224, 224, 3))) == 3


def test_health_endpoint(client):
    """The health endpoint reports the service is up."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
