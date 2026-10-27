const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `Lỗi API (${response.status})`);
  }
  return payload;
}

export const api = {
  get: (path) => request(path),
  postForm: (path, form) => request(path, { method: "POST", body: form }),
  postJson: (path, body) =>
    request(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  delete: (path) => request(path, { method: "DELETE" }),
  cameraStream: `${API_BASE}/api/camera/stream`,
};

export function imageForm(image, values = {}) {
  const form = new FormData();
  form.append("image", image);
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== null) form.append(key, String(value));
  });
  return form;
}
