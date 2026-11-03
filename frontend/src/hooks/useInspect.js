import { useState } from "react";
import { api, imageForm } from "../services/api.js";
import { useAppStore } from "../store/appStore.js";

export function useInspect(endpoint = "/api/inspect") {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const setGlobalError = useAppStore((state) => state.setError);

  const inspect = async (image, values = {}) => {
    if (!image) {
      setError("Vui lòng chọn ảnh PCB cần quét.");
      return;
    }
    setBusy(true);
    setError("");
    setGlobalError("");
    try {
      const response = await api.postForm(endpoint, imageForm(image, values));
      setResult(response);
      if (typeof response.is_ok === "boolean") {
        try {
          await api.sendConveyorResult(response.is_ok);
        } catch (sendError) {
          setError(
            `Kết quả đã hiển thị nhưng không gửi được đến ESP: ${sendError.message}`,
          );
        }
      }
      return response;
    } catch (requestError) {
      setError(requestError.message);
      return null;
    } finally {
      setBusy(false);
    }
  };

  return { result, setResult, error, setError, busy, inspect };
}
