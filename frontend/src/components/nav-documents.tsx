"use client"

import {
    FileText,
    MoreHorizontal,
    Trash2,
    File,
    CornerUpRight,
    Loader2,
} from "lucide-react"

import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuSeparator,
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

interface NavDocumentsProps {
    documents: {
        id: string
        name: string
        url: string
        date?: string
    }[]
    loading?: boolean
    onDelete?: (id: string) => Promise<void>
}

export function NavDocuments({ documents, loading = false, onDelete }: NavDocumentsProps) {
    const { isMobile } = useSidebar()

    const handleDelete = async (id: string) => {
        if (onDelete) {
            await onDelete(id)
        }
    }

    return (
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
            <SidebarGroupLabel>Documents</SidebarGroupLabel>
            <SidebarMenu>
                {loading ? (
                    <div className="flex items-center gap-2 px-2 py-2 text-xs text-muted-foreground">
                        <Loader2 className="size-3 animate-spin" />
                        <span>Loading documents...</span>
                    </div>
                ) : documents.length > 0 ? (
                    documents.map((item) => (
                        <SidebarMenuItem key={item.id}>
                            <SidebarMenuButton asChild>
                                <a href={item.url} title={item.name}>
                                    <FileText />
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
                                        <CornerUpRight className="text-muted-foreground" />
                                        <span>Open in Viewer</span>
                                    </DropdownMenuItem>
                                    <DropdownMenuSeparator />
                                    <DropdownMenuItem
                                        className="text-destructive focus:text-destructive"
                                        onClick={() => handleDelete(item.id)}
                                    >
                                        <Trash2 className="text-destructive" />
                                        <span>Delete Document</span>
                                    </DropdownMenuItem>
                                </DropdownMenuContent>
                            </DropdownMenu>
                        </SidebarMenuItem>
                    ))
                ) : (
                    <div className="px-2 text-xs text-muted-foreground">
                        No documents uploaded yet.
                    </div>
                )}
            </SidebarMenu>
        </SidebarGroup>
    )
}
