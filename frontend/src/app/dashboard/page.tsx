"use client"

import { AppSidebar } from '@/components/app-sidebar'
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Separator } from '@/components/ui/separator'
import {
    SidebarInset,
    SidebarProvider,
    SidebarTrigger,
} from '@/components/ui/sidebar'
import { Button } from "@/components/ui/button"
import { Send, Loader2 } from "lucide-react"
import { Textarea } from '@/components/ui/textarea'
import { useChat } from '@/hooks/use-chat'
import { useState, useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { BotMessageSkeleton } from '@/components/skeletons'

export default function Page() {
    const { messages, sendMessage, isTyping, error } = useChat()
    const [inputValue, setInputValue] = useState('')
    const [isSending, setIsSending] = useState(false)
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages, isTyping])

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        if (!inputValue.trim() || isSending) return

        // validation
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

        await sendMessage(messageContent)
        setIsSending(false)
    }

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e)
        }
    }

    return (
        <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
                <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
                    <div className="flex items-center gap-2 px-4">
                        <SidebarTrigger className="-ml-1" />
                        <Separator
                            orientation="vertical"
                            className="mr-2 data-[orientation=vertical]:h-4"
                        />
                        <Breadcrumb>
                            <BreadcrumbList>
                                <BreadcrumbItem className="hidden md:block">
                                    <BreadcrumbLink href="#">
                                        DocuCentric
                                    </BreadcrumbLink>
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
                    <div className="flex-1 overflow-y-auto p-4">
                        {messages.length === 0 ? (
                            <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
                                <p className="text-sm">No messages yet. Start a conversation!</p>
                            </div>
                        ) : (
                            <div className="mx-auto max-w-3xl space-y-4">
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
                                                "max-w-[80%] rounded-2xl px-4 py-3",
                                                message.role === 'user'
                                                    ? 'bg-primary text-primary-foreground'
                                                    : 'bg-muted'
                                            )}
                                        >
                                            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                                            <p className="mt-1 text-[10px] opacity-60">
                                                {new Date(message.timestamp).toLocaleTimeString()}
                                            </p>
                                        </div>
                                    </div>
                                ))}
                                {isTyping && (
                                    <BotMessageSkeleton />
                                )}
                                <div ref={messagesEndRef} />
                            </div>
                        )}
                    </div>
                    <div className="border-t bg-background p-4">
                        <form onSubmit={handleSubmit} className="relative mx-auto flex max-w-3xl items-end gap-2 rounded-xl bg-muted/50 p-2 shadow-sm focus-within:ring-1 focus-within:ring-ring">
                            <Textarea
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Type your message here..."
                                className="min-h-[44px] w-full resize-none border-0 bg-transparent p-3 shadow-none focus-visible:ring-0"
                                disabled={isSending}
                            />
                            <Button type="submit" size="icon" className="shrink-0" disabled={isSending || !inputValue.trim()}>
                                {isSending ? (
                                    <Loader2 className="size-4 animate-spin" />
                                ) : (
                                    <Send className="size-4" />
                                )}
                                <span className="sr-only">Send</span>
                            </Button>
                        </form>
                    </div>
                </div>
            </SidebarInset>
        </SidebarProvider>
    )
}

