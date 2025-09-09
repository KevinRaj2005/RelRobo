import React, { useState, useEffect, useRef } from "react";
import { X, Moon, History, MessageCircle, RefreshCcw } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export default function ChatWidget() {
    const [open, setOpen] = useState(false);
    const [dark, setDark] = useState(false);
    const [messages, setMessages] = useState([
        {
            role: "bot",
            text: "Good morning! 👋 I'm your SupportBot. Please select a category below to get started.",
        },
    ]);
    const [input, setInput] = useState("");
    const [typing, setTyping] = useState(null);
    const [categoriesVisible, setCategoriesVisible] = useState(true);
    const [history, setHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);
    const [activeCategory, setActiveCategory] = useState(null);
    const messagesEndRef = useRef(null);

    // Format backend responses
    const formatResponse = (answer) => {
        try {
            if (typeof answer === "string" && (answer.includes("{") || answer.includes("["))) {
                answer = JSON.parse(answer);
            }
        } catch {
            return answer;
        }

        if (typeof answer === "object" && answer.title && Array.isArray(answer.steps)) {
            return `**${answer.title}**\n\n${answer.steps.map((s, i) => `${i + 1}. ${s}`).join("\n")}`;
        }

        if (typeof answer === "object" && !Array.isArray(answer)) {
            return Object.entries(answer)
                .map(([key, val]) => `- **${key}:** ${val}`)
                .join("\n");
        }

        return typeof answer === "string" ? answer : JSON.stringify(answer, null, 2);
    };

    // Auto-scroll
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, typing]);

    // Save chat session when closing
    useEffect(() => {
        if (!open && messages.length > 1) {
            setHistory((prev) => [...prev, messages]);
        }
    }, [open]);

    const handleCategory = async (category) => {
        setActiveCategory(category);
        setMessages((prev) => [...prev, { role: "user", text: category }]);
        setCategoriesVisible(false);
        setTyping("bot");

        try {
            const res = await fetch("http://127.0.0.1:8000/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: category }),
            });
            const data = await res.json();

            setMessages((prev) => [
                ...prev,
                { role: "bot", text: formatResponse(data.reply) || "❌ Sorry, no response." },
            ]);
        } catch {
            setMessages((prev) => [
                ...prev,
                { role: "bot", text: "❌ System error. Please try again later." },
            ]);
        }
        setTyping(null);
    };

    const handleSend = async () => {
        if (!input.trim()) return;
        const userMessage = input;

        setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
        setInput("");
        setTyping("bot");

        try {
            const res = await fetch("http://127.0.0.1:8000/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMessage }),
            });
            const data = await res.json();

            setMessages((prev) => [
                ...prev,
                { role: "bot", text: formatResponse(data.reply) || "❌ Sorry, no response." },
            ]);
        } catch {
            setMessages((prev) => [
                ...prev,
                { role: "bot", text: "❌ System error. Please try again later." },
            ]);
        }
        setTyping(null);
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleSend();
        }
    };

    if (!open) {
        return (
            <div className="fixed bottom-5 right-5 flex items-center group">
                <button
                    onClick={() => setOpen(true)}
                    className="relative bg-blue-500 text-white p-4 rounded-full shadow-lg 
                        hover:bg-blue-600 transition-all duration-500 transform group-hover:-translate-x-3"
                >
                    <MessageCircle size={28} />
                    <span className="absolute inset-0 rounded-full bg-blue-400 opacity-30 animate-ping"></span>
                </button>
                <span
                    className="opacity-0 group-hover:opacity-100 transform translate-x-5 group-hover:translate-x-0 
                        transition-all duration-500 text-sm font-medium bg-white shadow-md px-3 py-1 rounded-lg ml-3"
                >
                    How may I assist you?
                </span>
            </div>
        );
    }

    return (
        <div
            className={`fixed bottom-20 right-5 w-96 h-[600px] rounded-2xl shadow-xl border flex flex-col ${dark ? "bg-gray-900 text-white" : "bg-white text-black"
                }`}
        >
            {/* Header */}
            <div className="flex justify-between items-center border-b px-4 py-3 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-t-2xl">
                <h2 className="font-bold text-lg">RelRobo</h2>
                <div className="flex gap-3">
                    <button
                        onClick={() => {
                            setCategoriesVisible(true);
                            setActiveCategory(null);
                            setMessages((prev) => [
                                ...prev,
                                { role: "bot", text: "🔄 Please choose a new support category." },
                            ]);
                        }}
                        title="Change Category"
                    >
                        <RefreshCcw size={18} />
                    </button>
                    <button onClick={() => setDark(!dark)}>
                        <Moon size={18} />
                    </button>
                    <button onClick={() => setShowHistory(!showHistory)}>
                        <History size={18} />
                    </button>
                    <button onClick={() => setOpen(false)}>
                        <X size={18} />
                    </button>
                </div>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3 flex flex-col">
                {messages.map((m, i) => (
                    <div
                        key={i}
                        className={`whitespace-pre-wrap px-4 py-2 rounded-2xl max-w-[75%] text-sm shadow-md ${m.role === "bot"
                            ? dark
                                ? "bg-gray-700 text-gray-100 self-start"
                                : "bg-gray-100 text-gray-800 self-start"
                            : dark
                                ? "bg-blue-600 text-white self-end ml-auto"
                                : "bg-blue-500 text-white self-end ml-auto"
                            }`}
                    >
                        <ReactMarkdown
                            children={m.text}
                            remarkPlugins={[remarkGfm]}
                            components={{
                                a: ({ node, ...props }) => (
                                    <a
                                        {...props}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-blue-600 underline hover:text-blue-800"
                                    />
                                ),
                            }}
                        />
                    </div>
                ))}

                {typing && (
                    <div
                        className={`px-4 py-2 rounded-2xl max-w-[40%] text-sm shadow-md flex gap-1 ${typing === "bot"
                            ? dark
                                ? "bg-gray-700 text-gray-200 self-start"
                                : "bg-gray-200 text-gray-700 self-start"
                            : "bg-blue-400 text-white self-end ml-auto"
                            }`}
                    >
                        <span className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:0ms]" />
                        <span className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:150ms]" />
                        <span className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:300ms]" />
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Categories */}
            {categoriesVisible && (
                <div className="px-4 space-y-2">
                    {["IT Support", "Admin Support", "HR Support", "Human Support"].map((cat) => (
                        <button
                            key={cat}
                            className="w-full border rounded-lg p-2 hover:bg-blue-100 text-sm"
                            onClick={() => handleCategory(cat)}
                        >
                            {cat}
                        </button>
                    ))}
                </div>
            )}

            {/* Input */}
            <div className="flex items-center gap-2 px-4 py-3 border-t">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => {
                        setInput(e.target.value);
                        setTyping("user");
                    }}
                    onBlur={() => setTyping(null)}
                    onKeyDown={handleKeyDown}
                    placeholder="Type your message..."
                    className={`flex-1 border rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 ${dark ? "bg-gray-800 text-white placeholder-gray-400" : ""
                        }`}
                />
                <button
                    onClick={handleSend}
                    className="bg-blue-500 text-white p-2 rounded-lg hover:bg-blue-600"
                >
                    ➤
                </button>
            </div>

            {/* History Panel */}
            {showHistory && (
                <div
                    className={`absolute top-0 right-0 w-72 h-full border-l shadow-lg overflow-y-auto ${dark ? "bg-gray-800 text-white" : "bg-gray-100 text-black"
                        }`}
                >
                    <div className="p-3 font-semibold border-b flex justify-between items-center">
                        <span>Chat History</span>
                        <button
                            onClick={() => setShowHistory(false)}
                            className="text-xs bg-gray-300 px-2 py-1 rounded"
                        >
                            Back
                        </button>
                    </div>
                    {history.length === 0 ? (
                        <div className="p-3 text-sm text-gray-500">No past chats yet.</div>
                    ) : (
                        history.map((session, idx) => (
                            <div
                                key={idx}
                                className="p-3 border-b cursor-pointer hover:bg-blue-100"
                                onClick={() => {
                                    setMessages(session);
                                    setShowHistory(false);
                                }}
                            >
                                <p className="text-sm truncate">
                                    {session.find((m) => m.role === "user")?.text || "Conversation"}
                                </p>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
}
