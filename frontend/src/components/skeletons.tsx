"use client"

import { Skeleton } from "@/components/ui/skeleton"

export function NavItemSkeleton() {
    return (
        <div className="flex items-center gap-2 px-2 py-1.5">
            <Skeleton className="h-4 w-4 rounded-md" />
            <Skeleton className="h-4 flex-1 rounded-md" />
        </div>
    )
}

export function NavHistorySkeleton() {
    return (
        <div className="space-y-2 px-2">
            <NavItemSkeleton />
            <div className="flex items-center gap-2 px-2 py-1.5">
                <Skeleton className="h-4 w-4 rounded-md" />
                <Skeleton className="h-4 w-3/4 rounded-md" />
            </div>
            <div className="flex items-center gap-2 px-2 py-1.5">
                <Skeleton className="h-4 w-4 rounded-md" />
                <Skeleton className="h-4 w-1/2 rounded-md" />
            </div>
        </div>
    )
}

export function BotMessageSkeleton() {
    return (
        <div className="flex gap-3 justify-start w-full animate-in fade-in slide-in-from-bottom-2 duration-300">
            <div className="rounded-2xl bg-muted px-4 py-3 flex items-center gap-1">
                <div className="h-2 w-2 rounded-full bg-foreground/40 animate-bounce [animation-delay:-0.3s]" />
                <div className="h-2 w-2 rounded-full bg-foreground/40 animate-bounce [animation-delay:-0.15s]" />
                <div className="h-2 w-2 rounded-full bg-foreground/40 animate-bounce" />
            </div>
        </div>
    )
}

export function ChatSkeleton() {
    return (
        <div className="mx-auto max-w-3xl space-y-4 p-4">
            {/* Bot message skeleton */}
            <div className="flex gap-3 justify-start w-full">
                <div className="max-w-[80%] rounded-2xl bg-muted px-4 py-3 space-y-2">
                    <Skeleton className="h-4 w-[250px] bg-foreground/10" />
                    <Skeleton className="h-4 w-[180px] bg-foreground/10" />
                </div>
            </div>

            {/* User message skeleton */}
            <div className="flex gap-3 justify-end w-full">
                <div className="max-w-[80%] rounded-2xl bg-primary/20 px-4 py-3 space-y-2">
                    <Skeleton className="h-4 w-[200px] bg-background/20" />
                </div>
            </div>

            {/* Bot message skeleton */}
            <BotMessageSkeleton />
        </div>
    )
}
