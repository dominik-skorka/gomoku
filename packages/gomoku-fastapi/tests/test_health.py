"""``GET /healthz`` is the simplest possible smoke test."""


class TestHealth:
    def test_returns_ok(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_no_authentication_required(self, client):
        # The probe must work without any headers — it's called by
        # orchestrators (k8s, ECS, …) before any auth middleware is
        # configured, and silently 401-ing would mark the pod
        # unhealthy.
        response = client.get("/healthz", headers={})
        assert response.status_code == 200
