import { cn } from "@/lib/utils";
import { CSSProperties, FC, ReactNode } from "react";

interface ShinyTextProps {
    children: ReactNode;
    disabled?: boolean;
    speed?: number;
    className?: string;
    shimmerWidth?: number;
}

const ShinyText: FC<ShinyTextProps> = ({
    children,
    disabled = false,
    speed = 5,
    className,
    shimmerWidth = 100,
}) => {
    const animationDuration = `${speed}s`;

    return (
        <div
            className={cn(
                "bg-clip-text text-transparent bg-gradient-to-r from-transparent via-black/80 via-50% to-transparent  dark:via-white/80",
                "bg-[length:200%_100%] animate-shimmer",
                disabled ? "" : "animate-shimmer",
                className
            )}
            style={
                {
                    "--shimmer-width": `${shimmerWidth}px`,
                    animationDuration: animationDuration,
                } as CSSProperties
            }
        >
            {children}
        </div>
    );
};

export default ShinyText;
