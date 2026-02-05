"use client"

import { Activity, CheckCircle2, Zap } from "lucide-react"

import {
    SidebarGroup,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
} from "@/components/ui/sidebar"

export function NavSystemStatus() {
    return (
        <SidebarGroup className="mt-auto">
            <SidebarGroupLabel>System Status</SidebarGroupLabel>
            <SidebarMenu>
                <SidebarMenuItem>
                    <div className="flex flex-col gap-2 p-2">
                        <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-2 text-muted-foreground">
                                <Activity className="size-3" />
                                Vector Store
                            </span>
                            <span className="flex items-center gap-1 text-emerald-500 font-medium">
                                <CheckCircle2 className="size-3" />
                                Online
                            </span>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                            <span className="flex items-center gap-2 text-muted-foreground">
                                <Zap className="size-3" />
                                Model
                            </span>
                            <span className="font-medium text-sidebar-foreground">
                                BART-MNLI
                            </span>
                        </div>
                    </div>
                </SidebarMenuItem>
            </SidebarMenu>
        </SidebarGroup>
    )
}
