import RetroGrid from "@/components/backgrounds/RetroGrid";
import ShinyText from "@/components/ui/ShinyText"; // Import ShinyText from correct location
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  return (
    <div className="relative flex h-screen w-full flex-col items-center justify-center overflow-hidden rounded-lg border bg-background md:shadow-xl">
      <div className="z-10 flex flex-col items-center justify-center space-y-4 px-4 text-center -mt-32">
        <ShinyText
          className="text-6xl font-bold tracking-tighter text-transparent md:text-8xl lg:text-9xl"
          shimmerWidth={200}
          speed={3}
        >
          <span className="font-space-grotesk uppercase">DocuCentric</span>
        </ShinyText>

        <div className="max-w-2xl">
          <p className="text-xl text-muted-foreground md:text-2xl font-light tracking-wide uppercase">
            Enterprise AI Document Intelligence <br />
            <span className="font-semibold text-foreground/80">Reasoning Platform</span>
          </p>
        </div>

        <div className="pt-8">
          <Link href="/dashboard">
            <Button variant="cool" size="xl">
              Launch Platform
            </Button>
          </Link>
        </div>
      </div>

      <RetroGrid />
    </div>
  );
}
