import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // LAN内のスマホ等から開発サーバーへアクセスできるようにする（IPは .env.local の DEV_LAN_HOST で指定）
  allowedDevOrigins: process.env.DEV_LAN_HOST ? [process.env.DEV_LAN_HOST] : undefined,
};

export default nextConfig;
