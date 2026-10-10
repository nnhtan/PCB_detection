import { useCallback, useEffect, useState } from "react";
import { api } from "../services/api.js";

export function useCamera() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const payload = await api.get("/api/camera/status");
      setStatus(payload.camera);
    } catch (requestError) {
      setError(requestError.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 2500);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const connect = async (source) => {
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("source", source);
      const payload = await api.postForm("/api/camera/connect", form);
      setStatus(payload.camera);
      if (payload.status !== "success") throw new Error(payload.message);
    } catch (requestError) {
      setError(requestError.message);
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const disconnect = async () => {
    setBusy(true);
    try {
      const payload = await api.postForm("/api/camera/disconnect", new FormData());
      setStatus(payload.camera);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  return { status, error, busy, connect, disconnect, refresh };
}
