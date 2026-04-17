"use client"

import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Separator } from '@/components/ui/separator'
import { SidebarTrigger } from '@/components/ui/sidebar'
import { Button } from "@/components/ui/button"
import { Send, Loader2, FileText, X, Plus } from "lucide-react"
import { Textarea } from '@/components/ui/textarea'
import { useChat } from '@/hooks/use-chat'
import { useDocuments } from '@/hooks/use-documents'
import { useState, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { BotMessageSkeleton } from '@/components/skeletons'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export default function DashboardPage() {
    const { messages, sendMessage, isTyping, error, sessionId, clearSession } = useChat()
    const { documents } = useDocuments()
    const [inputValue, setInputValue] = useState('')
    const [isSending, setIsSending] = useState(false)
    const [selectedDocs, setSelectedDocs] = useState<string[]>([])
    const [showDocSelector, setShowDocSelector] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages, isTyping])

    const toggleDocument = (docId: string) => {
        setSelectedDocs(prev =>
            prev.includes(docId)
                ? prev.filter(id => id !== docId)
                : [...prev, docId]
        )
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        if (!inputValue.trim() || isSending) return

        try {
            const { ChatInputSchema } = require("@/lib/validation")
            ChatInputSchema.parse({ message: inputValue })
        } catch (err: any) {
            const { toast } = require("sonner")
            toast.error(err.errors?.[0]?.message || "Invalid message")
            return
        }

        const messageContent = inputValue.trim()
        setInputValue('')
        setIsSending(true)

        await sendMessage(messageContent, selectedDocs.length > 0 ? selectedDocs : undefined)
        setIsSending(false)
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
        }
    }

    const selectedDocumentNames = documents
        ?.filter(d => selectedDocs.includes(d.id))
        .map(d => d.filename || d.originalName) || []

    return (
        <>
            <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
                <div className="flex items-center gap-2 px-4">
                    <SidebarTrigger className="-ml-1" />
                    <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
                    <Breadcrumb>
                        <BreadcrumbList>
                            <BreadcrumbItem className="hidden md:block">
                                <BreadcrumbLink href="#">DocuCentric</BreadcrumbLink>
                            </BreadcrumbItem>
                            <BreadcrumbSeparator className="hidden md:block" />
                            <BreadcrumbItem>
                                <BreadcrumbPage>Dashboard</BreadcrumbPage>
                            </BreadcrumbItem>
                        </BreadcrumbList>
                    </Breadcrumb>
                </div>
            </header>

            <div className="flex flex-1 flex-col overflow-hidden">
                {/* Document Selector Bar */}
                {documents && documents.length > 0 && (
                    <div className="border-b bg-muted/30 p-3">
                        <div className="mx-auto max-w-4xl">
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                    <FileText className="size-4 text-muted-foreground" />
                                    <span className="text-sm font-medium">
                                        {selectedDocs.length === 0
                                            ? "Select documents to analyze"
                                            : `${selectedDocs.length} document(s) selected`}
                                    </span>
                                </div>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => setShowDocSelector(!showDocSelector)}
                                >
                                    <Plus className="size-3 mr-1" />
                                    {showDocSelector ? "Hide" : "Add"}
                                </Button>
                            </div>

                            {/* Selected Documents */}
                            {selectedDocs.length > 0 && (
                                <div className="flex flex-wrap gap-2 mb-2">
                                    {selectedDocumentNames.map((name, idx) => (
                                        <Badge key={idx} variant="secondary" className="gap-1">
                                            <FileText className="size-3" />
                                            <span className="max-w-[200px] truncate">{name}</span>
                                            <X
                                                className="size-3 ml-1 cursor-pointer hover:text-destructive"
                                                onClick={() => toggleDocument(selectedDocs[idx])}
                                            />
                                        </Badge>
                                    ))}
                                </div>
                            )}

                            {/* Document Selector Dropdown */}
                            {showDocSelector && (
                                <Card className="p-3 max-h-48 overflow-y-auto">
                                    <div className="space-y-2">
                                        {documents.map((doc) => (
                                            <label
                                                key={doc.id}
                                                className="flex items-center gap-2 cursor-pointer hover:bg-muted p-2 rounded"
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={selectedDocs.includes(doc.id)}
                                                    onChange={() => toggleDocument(doc.id)}
                                                    className="size-4"
                                                />
                                                <FileText className="size-4 text-muted-foreground" />
                                                <span className="text-sm flex-1 truncate">
                                                    {doc.filename || doc.originalName}
                                                </span>
                                                <span className="text-xs text-muted-foreground">
                                                    {doc.status}
                                                </span>
                                            </label>
                                        ))}
                                    </div>
                                </Card>
                            )}
                        </div>
                    </div>
                )}

                {/* Messages Area */}
                <div className="flex-1 overflow-y-auto p-4">
                    {messages.length === 0 ? (
                        <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
                            <div className="text-6xl">💬</div>
                            <h3 className="text-lg font-medium">Start Your Document Analysis</h3>
                            <p className="text-sm text-center max-w-md">
                                {documents && documents.length > 0
                                    ? `You have ${documents.length} document(s) ready. Select which ones above, then ask me anything!`
                                    : "Upload some documents first, then I'll help you analyze them."}
                            </p>
                            {documents && documents.length > 0 && (
                                <p className="text-xs text-muted-foreground/60">
                                    💡 Pro tip: Select multiple documents to compare and cross-reference them!
                                </p>
                            )}
                        </div>
                    ) : (
                        <div className="mx-auto max-w-4xl space-y-4">
                            {messages.map((message) => (
                                <div
                                    key={message.id}
                                    className={cn(
                                        "flex gap-3",
                                        message.role === 'user' ? 'justify-end' : 'justify-start'
                                    )}
                                >
                                    <div
                                        className={cn(
                                            "max-w-[85%] rounded-2xl px-5 py-4",
                                            message.role === 'user'
                                                ? 'bg-primary text-primary-foreground'
                                                : 'bg-muted'
                                        )}
                                    >
                                        {/* Reasoning Steps */}
                                        {message.reasoning && message.reasoning.length > 0 && (
                                            <div className="mb-3 space-y-1.5 border-b border-black/5 pb-3 dark:border-white/5">
                                                {message.reasoning.map((step, i) => (
                                                    <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground/80">
                                                        <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary/40" />
                                                        <span>{step}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}

                                        {/* Message Content */}
                                        <div className="text-sm whitespace-pre-wrap leading-relaxed">
                                            {message.content}
                                        </div>

                                        {/* Metadata */}
                                        <div className="mt-2 flex items-center justify-between text-[10px] opacity-60">
                                            <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
                                            {message.documentContext && (
                                                <span>
                                                    {message.documentContext.numDocuments || 1} doc(s) analyzed
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {isTyping && <BotMessageSkeleton />}
                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Input Area */}
                <div className="border-t bg-background p-4">
                    <form onSubmit={handleSubmit} className="relative mx-auto flex max-w-4xl items-end gap-2 rounded-xl bg-muted/50 p-2 shadow-sm focus-within:ring-2 focus-within:ring-primary/20">
                        <Textarea
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder={
                                selectedDocs.length > 0
                                    ? `Ask about ${selectedDocs.length} document(s)...`
                                    : "Type your question about the documents..."
                            }
                            className="min-h-[44px] max-h-[120px] w-full resize-none border-0 bg-transparent p-3 shadow-none focus-visible:ring-0"
                            disabled={isSending}
                            rows={1}
                        />
                        <Button
                            type="submit"
                            size="icon"
                            className="shrink-0 h-10 w-10"
                            disabled={isSending || !inputValue.trim()}
                        >
                            {isSending ? (
                                <Loader2 className="size-4 animate-spin" />
                            ) : (
                                <Send className="size-4" />
                            )}
                            <span className="sr-only">Send</span>
                        </Button>
                    </form>
                    {error && (
                        <p className="text-xs text-destructive mt-2 text-center">{error}</p>
                    )}
                </div>
            </div>
        </>
    )
}
