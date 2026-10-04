import { useEffect, useState } from "react";
import { api } from "./api.js";

/** Polls GET /api/health (every 15 s) so the top bar and the System page show the real model status. */
export function useHealth() {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let alive = true;
    const load = () =>
      api.health()
        .then((data) => alive && setState({ loading: false, data, error: null }))
        .catch((e) => alive && setState({ loading: false, data: null, error: e.message }));
    load();
    const id = setInterval(load, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return state;
}
