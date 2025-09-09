import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
});

export const askQuery = async (query) => {
  const res = await API.post("/ask", { query });
  return res.data;
};

export const getEmployees = async () => {
  const res = await API.get("/employee");
  return res.data;
};
