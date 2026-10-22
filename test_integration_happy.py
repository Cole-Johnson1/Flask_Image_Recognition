"""Integration happy-path test using a real sample image."""


def test_successful_prediction(client):
    """A real hand sign image goes through the full pipeline and returns a digit."""
    with open("test_images/3/Sign 3 (30).jpeg", "rb") as handle:
        response = client.post(
            "/prediction",
            data={"file": (handle, "sign.jpeg")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert b"File cannot be processed" not in response.data
