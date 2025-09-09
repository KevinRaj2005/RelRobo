import React, { useState } from "react";
import { askQuery } from "./api";
import {
  PaperAirplaneIcon,
  ExclamationTriangleIcon,
  TrashIcon,
  EllipsisVerticalIcon,
  MoonIcon,
  SunIcon,
} from "@heroicons/react/24/outline";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  const categories = ["IT Support", "Admin Support", "HR Support"];

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await askQuery(input);
      setMessages((prev) => [
        ...prev,
        { text: response.answer || "No response", sender: "bot" },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { text: "⚠️ Error: could not fetch response", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`flex flex-col h-full ${darkMode ? "bg-gray-900 text-white" : "bg-white text-black"}`}>

      {/* Header */}
      <div className="flex justify-between items-center p-3 bg-gradient-to-r from-purple-600 to-blue-500 text-white">
        <div>
          <h1 className="font-bold">AssistRA</h1>
          <p className="text-xs opacity-90">AI-powered support assistant</p>
        </div>
        <div className="flex gap-2">
          <button onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
          </button>
          <button onClick={() => setMessages([])}>
            <TrashIcon className="w-5 h-5" />
          </button>
          <button>
            <EllipsisVerticalIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {/* Greeting card */}
        {messages.length === 0 && (
          <div className="bg-white shadow rounded-xl p-3 border">
            <p className="font-semibold">Good afternoon! 👋</p>
            <p className="text-sm text-gray-600">
              I'm your SupportBot. Please select a category below to get started.
            </p>
          </div>
        )}

        {/* Category buttons */}
        {messages.length === 0 && (
          <div className="flex flex-col gap-2 mt-3">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setMessages((prev) => [...prev, { text: cat, sender: "user" }])}
                className="border rounded-lg p-2 text-left hover:bg-gray-100 transition"
              >
                {cat}
              </button>
            ))}
          </div>
        )}

        {/* Chat bubbles */}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-2 rounded-lg max-w-[75%] ${msg.sender === "user"
                ? "bg-blue-500 text-white self-end ml-auto"
                : "bg-gray-200 text-black self-start mr-auto"
              }`}
          >
            {msg.text}
          </div>
        ))}

        {loading && (
          <div className="flex items-center space-x-2 text-gray-500">
            <ExclamationTriangleIcon className="w-4 h-4" />
            <span>Thinking...</span>
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="p-3 border-t flex items-center gap-2">
        <input
          className="flex-1 border rounded-full p-2 px-4 focus:outline-none"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask me anything..."
        />
        <button className="p-2 bg-blue-500 text-white rounded-full" onClick={sendMessage} disabled={loading}>
          <PaperAirplaneIcon className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}
