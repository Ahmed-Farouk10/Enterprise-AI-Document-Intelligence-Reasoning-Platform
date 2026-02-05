"use client"

import { MessageSquare, MoreHorizontal, Trash2 } from "lucide-react"

import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
    SidebarGroup,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuAction,
    SidebarMenuButton,
    SidebarMenuItem,
    useSidebar,
} from "@/components/ui/sidebar"

export function NavHistory({
    chats,
    loading = false,
}: {
    chats: {
        name: string
        url: string
        date: string
    }[]
    loading?: boolean
}) {
    const { isMobile } = useSidebar()

    // Import here to avoid circular dependencies if any
    const { NavHistorySkeleton } = require("@/components/skeletons")

    return (
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
            <SidebarGroupLabel>Chat History</SidebarGroupLabel>
            <SidebarMenu>
                {loading ? (
                    <NavHistorySkeleton />
                ) : chats.length === 0 ? (
                    <SidebarMenuItem>
                        <SidebarMenuButton disabled>
                            <span>No past chats</span>
                        </SidebarMenuButton>
                    </SidebarMenuItem>
                ) : (
                    chats.map((item) => (
                        <SidebarMenuItem key={item.name}>
                            <SidebarMenuButton asChild>
                                <a href={item.url}>
                                    <MessageSquare />
                                    <span className="truncate">{item.name}</span>
                                </a>
                            </SidebarMenuButton>
                            <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                    <SidebarMenuAction showOnHover>
                                        <MoreHorizontal />
                                        <span className="sr-only">More</span>
                                    </SidebarMenuAction>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent
                                    className="w-48 rounded-lg"
                                    side={isMobile ? "bottom" : "right"}
                                    align={isMobile ? "end" : "start"}
                                >
                                    <DropdownMenuItem>
                                        <Trash2 className="text-muted-foreground" />
                                        <span>Delete Chat</span>
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>
                        </SidebarMenuItem>
                    ))
                )}
                {chats.length > 0 && (
                    <SidebarMenuItem>
                        <SidebarMenuButton className="text-sidebar-foreground/70">
                            <MoreHorizontal className="text-sidebar-foreground/70" />
                            <span>View All History</span>
                        </SidebarMenuButton>
                    </SidebarMenuItem>
                )}
            </SidebarMenu>
        </SidebarGroup>
    )
}
