"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { ZoomIn, ZoomOut, Maximize2, Minimize2, Network } from "lucide-react"

interface Node {
    id: string
    label: string
    type: string
    x?: number
    y?: number
}

interface Edge {
    source: string
    target: string
    label: string
}

export function KnowledgeGraphVisualization() {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const [zoom, setZoom] = useState(1)
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [nodes, setNodes] = useState<Node[]>([])
    const [edges, setEdges] = useState<Edge[]>([])

    useEffect(() => {
        const fetchGraphData = async () => {
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
                const response = await fetch(`${apiUrl}/api/graph/nodes?limit=50`)

                if (!response.ok) {
                    throw new Error('Failed to fetch graph data')
                }

                const data = await response.json()

                // Transform API data to component format
                const transformedNodes: Node[] = data.nodes.map((node: any) => ({
                    id: node.id,
                    label: node.label,
                    type: node.type
                }))

                const transformedEdges: Edge[] = data.edges.map((edge: any) => ({
                    source: edge.source,
                    target: edge.target,
                    label: edge.label
                }))

                setNodes(transformedNodes)
                setEdges(transformedEdges)
            } catch (error) {
                console.error('Failed to fetch graph data:', error)
                // Keep empty arrays on error
            }
        }

        fetchGraphData()
    }, [])

    useEffect(() => {
        const canvas = canvasRef.current
        const container = containerRef.current
        if (!canvas || !container) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Set canvas size
        canvas.width = container.clientWidth
        canvas.height = container.clientHeight

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Apply zoom
        ctx.save()
        ctx.scale(zoom, zoom)

        // Calculate node positions (simple force-directed layout simulation)
        const centerX = canvas.width / (2 * zoom)
        const centerY = canvas.height / (2 * zoom)
        const radius = Math.min(centerX, centerY) * 0.6

        nodes.forEach((node, i) => {
            const angle = (i / nodes.length) * 2 * Math.PI
            node.x = centerX + radius * Math.cos(angle)
            node.y = centerY + radius * Math.sin(angle)
        })

        // Draw edges
        ctx.strokeStyle = 'hsl(var(--muted-foreground) / 0.3)'
        ctx.lineWidth = 2
        edges.forEach(edge => {
            const source = nodes.find(n => n.id === edge.source)
            const target = nodes.find(n => n.id === edge.target)
            if (source && target && source.x && source.y && target.x && target.y) {
                ctx.beginPath()
                ctx.moveTo(source.x, source.y)
                ctx.lineTo(target.x, target.y)
                ctx.stroke()

                // Draw edge label
                const midX = (source.x + target.x) / 2
                const midY = (source.y + target.y) / 2
                ctx.fillStyle = 'hsl(var(--muted-foreground))'
                ctx.font = '10px sans-serif'
                ctx.textAlign = 'center'
                ctx.fillText(edge.label, midX, midY)
            }
        })

        // Draw nodes
        nodes.forEach(node => {
            if (!node.x || !node.y) return

            // Node circle
            ctx.beginPath()
            ctx.arc(node.x, node.y, 30, 0, 2 * Math.PI)

            // Color based on type
            if (node.type === 'document') {
                ctx.fillStyle = 'hsl(var(--primary))'
            } else if (node.type === 'entity') {
                ctx.fillStyle = 'hsl(var(--chart-2))'
            } else {
                ctx.fillStyle = 'hsl(var(--chart-3))'
            }
            ctx.fill()
            ctx.strokeStyle = 'hsl(var(--background))'
            ctx.lineWidth = 3
            ctx.stroke()

            // Node label
            ctx.fillStyle = 'hsl(var(--foreground))'
            ctx.font = 'bold 12px sans-serif'
            ctx.textAlign = 'center'
            ctx.fillText(node.label, node.x, node.y + 50)
        })

        ctx.restore()
    }, [nodes, edges, zoom])

    const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.2, 3))
    const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.2, 0.5))
    const handleResetZoom = () => setZoom(1)
    const toggleFullscreen = () => setIsFullscreen(!isFullscreen)

    return (
        <div ref={containerRef} className="relative h-full w-full">
            {/* Controls */}
            <div className="absolute right-4 top-4 z-10 flex flex-col gap-2">
                <Button
                    variant="outline"
                    size="icon"
                    onClick={handleZoomIn}
                    className="bg-background/95 backdrop-blur"
                >
                    <ZoomIn className="size-4" />
                </Button>
                <Button
                    variant="outline"
                    size="icon"
                    onClick={handleZoomOut}
                    className="bg-background/95 backdrop-blur"
                >
                    <ZoomOut className="size-4" />
                </Button>
                <Button
                    variant="outline"
                    size="icon"
                    onClick={handleResetZoom}
                    className="bg-background/95 backdrop-blur"
                >
                    <span className="text-xs font-bold">1:1</span>
                </Button>
                <Button
                    variant="outline"
                    size="icon"
                    onClick={toggleFullscreen}
                    className="bg-background/95 backdrop-blur"
                >
                    {isFullscreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
                </Button>
            </div>

            {/* Legend */}
            <div className="absolute bottom-4 left-4 z-10 rounded-lg border bg-background/95 p-3 backdrop-blur">
                <p className="mb-2 text-xs font-semibold">Legend</p>
                <div className="space-y-1.5">
                    <div className="flex items-center gap-2">
                        <div className="size-3 rounded-full bg-primary" />
                        <span className="text-xs">Documents</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="size-3 rounded-full bg-chart-2" />
                        <span className="text-xs">Entities</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="size-3 rounded-full bg-chart-3" />
                        <span className="text-xs">Concepts</span>
                    </div>
                </div>
            </div>

            {/* Canvas */}
            <canvas
                ref={canvasRef}
                className="h-full w-full"
            />

            {/* Empty State */}
            {nodes.length === 0 && (
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                        <Network className="mx-auto mb-2 size-12 opacity-20" />
                        <p className="text-sm">No graph data available</p>
                        <p className="text-xs">Upload documents to build the knowledge graph</p>
                    </div>
                </div>
            )}
        </div>
    )
}
