import { create } from "zustand";
import { api } from "../services/api.js";

export const useAppStore = create((set) => ({
  info: null,
  reference: null,
  busy: false,
  error: "",
  setBusy: (busy) => set({ busy }),
  setError: (error) => set({ error }),
  loadSystem: async () => {
    try {
      const [info, reference] = await Promise.all([
        api.get("/api/info"),
        api.get("/api/reference"),
      ]);
      set({ info, reference, error: "" });
    } catch (error) {
      set({ error: error.message });
    }
  },
  loadReference: async () => {
    const reference = await api.get("/api/reference");
    set({ reference });
    return reference;
  },
}));
