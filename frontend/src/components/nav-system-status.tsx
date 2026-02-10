"use client"

import { Network, Activity, Database, Cpu } from "lucide-react"
import Link from "next/link"
import {
    SidebarGroup,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuItem,
    SidebarMenuButton,
} from "@/components/ui/sidebar"

export function NavSystemStatus() {
    return (
        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
            <SidebarGroupLabel>System</SidebarGroupLabel>
            <SidebarMenu>
                <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                        <Link href="/knowledge-graph">
                            <Network className="size-4" />
                            <span>Knowledge Graph</span>
                        </Link>
                    </SidebarMenuButton>
                </SidebarMenuItem>
                <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                        <Link href="/dashboard">
                            <Activity className="size-4" />
                            <span>Dashboard</span>
                        </Link>
                    </SidebarMenuButton>
                </SidebarMenuItem>
            </SidebarMenu>
        </SidebarGroup>
    )
}
