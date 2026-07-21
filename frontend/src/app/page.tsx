"use client";

import { useEffect, useRef, useState } from "react";
import type { PredictResult } from "./types";

// NEXT_PUBLIC_API_URL が未設定の場合、アクセス元ホスト(localhost / LAN IP)に合わせてバックエンドのURLを自動決定する
function getApiUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined")
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  return "http://localhost:8000";
}

type Step = "input" | "result";
type Tab = "camera" | "upload";

export default function Home() {
  const [step, setStep] = useState<Step>("input");
  const [tab, setTab] = useState<Tab>("camera");
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoReady, setVideoReady] = useState(false);

  const uploadInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraOn(false);
    setVideoReady(false);
  };

  useEffect(() => stopCamera, []);

  // <video> がマウントされてからストリームを紐付ける（起動直後は videoRef がまだ null なため）
  useEffect(() => {
    if (!cameraOn || !videoRef.current || !streamRef.current) return;
    videoRef.current.srcObject = streamRef.current;
    videoRef.current.play().catch(() => {
      setCameraError("カメラ映像の再生に失敗しました。");
    });
  }, [cameraOn]);

  const startCamera = async () => {
    setCameraError(null);
    setVideoReady(false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      setCameraOn(true);
    } catch {
      setCameraError(
        "カメラを起動できませんでした。ブラウザのカメラ権限を確認してください。"
      );
    }
  };

  const capturePhoto = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    if (!video.videoWidth || !video.videoHeight) {
      setCameraError(
        "カメラの映像がまだ準備できていません。少し待ってからもう一度お試しください。"
      );
      return;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) {
          handleSelectFile(new File([blob], "camera.jpg", { type: "image/jpeg" }));
        }
        stopCamera();
      },
      "image/jpeg",
      0.92
    );
  };

  const handleSelectFile = (file: File | undefined) => {
    if (!file) return;
    if (imgPreview) URL.revokeObjectURL(imgPreview);
    setImgFile(file);
    setImgPreview(URL.createObjectURL(file));
  };

  const handleReset = () => {
    stopCamera();
    setCameraError(null);
    if (imgPreview) URL.revokeObjectURL(imgPreview);
    setStep("input");
    setImgFile(null);
    setImgPreview(null);
    setResult(null);
    setError(null);
    if (uploadInputRef.current) uploadInputRef.current.value = "";
  };

  const runPredict = async () => {
    if (!imgFile) return;
    setStep("result");
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", imgFile);
      const res = await fetch(`${getApiUrl()}/predict`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error("解析に失敗しました。");
      const data: PredictResult = await res.json();
      setResult(data);
    } catch {
      setError("解析に失敗しました。もう一度お試しください。");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main
      className="min-h-screen bg-cover bg-center bg-fixed flex items-center justify-center py-10 px-4"
      style={{ backgroundImage: "url('/background.png')" }}
    >
      <div className="w-full max-w-xl rounded-[25px] bg-black/40 backdrop-blur-lg p-8 text-white">
        {step === "input" && (
          <>
            <h1
              className="text-3xl font-bold mb-2"
              style={{ textShadow: "2px 2px 8px rgba(0,0,0,0.8)" }}
            >
              もぐもぐスキャナー
            </h1>
            <p className="mb-6 text-white/90">
              料理を撮影するか、画像をアップロードしてください。
            </p>

            <div className="flex gap-2 mb-4">
              <button
                onClick={() => setTab("camera")}
                className={`flex-1 py-2 rounded-full font-bold ${
                  tab === "camera" ? "bg-white/20" : "bg-white/5"
                }`}
              >
                カメラで撮影
              </button>
              <button
                onClick={() => {
                  stopCamera();
                  setTab("upload");
                }}
                className={`flex-1 py-2 rounded-full font-bold ${
                  tab === "upload" ? "bg-white/20" : "bg-white/5"
                }`}
              >
                画像をアップロード
              </button>
            </div>

            {tab === "camera" && (
              <div>
                {!cameraOn && (
                  <button
                    onClick={startCamera}
                    className="w-full py-3 rounded-full font-bold bg-white/10 border border-white/20"
                  >
                    カメラ起動
                  </button>
                )}
                {cameraError && (
                  <p className="text-red-400 text-sm mt-2">{cameraError}</p>
                )}
                {cameraOn && (
                  <div className="mt-2">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      onLoadedMetadata={() => setVideoReady(true)}
                      className="w-full rounded-2xl mb-3"
                    />
                    <button
                      onClick={capturePhoto}
                      disabled={!videoReady}
                      className="w-full h-14 rounded-full font-bold text-white disabled:opacity-50"
                      style={{
                        background:
                          "linear-gradient(135deg, #00dbde 0%, #fc00ff 100%)",
                      }}
                    >
                      {videoReady ? "📸 撮影する" : "カメラ準備中..."}
                    </button>
                    <button
                      onClick={stopCamera}
                      className="w-full py-2 mt-2 text-sm text-white/70"
                    >
                      キャンセル
                    </button>
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>
            )}

            {tab === "upload" && (
              <div>
                <input
                  ref={uploadInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/jpg,image/webp"
                  className="hidden"
                  onChange={(e) => handleSelectFile(e.target.files?.[0])}
                />
                <button
                  onClick={() => uploadInputRef.current?.click()}
                  className="w-full py-3 rounded-full font-bold bg-white/10 border border-white/20"
                >
                  ファイルを選択
                </button>
              </div>
            )}

            {imgPreview && (
              <div className="mt-6">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imgPreview}
                  alt="選択した画像"
                  className="w-full rounded-2xl mb-4"
                />
                <button
                  onClick={runPredict}
                  className="w-full h-14 rounded-full font-bold text-white"
                  style={{
                    background:
                      "linear-gradient(135deg, #00dbde 0%, #fc00ff 100%)",
                  }}
                >
                  {tab === "camera" ? "この写真で解析する" : "アップロード画像で解析"}
                </button>
              </div>
            )}
          </>
        )}

        {step === "result" && (
          <>
            <h1
              className="text-3xl font-bold mb-6"
              style={{ textShadow: "2px 2px 8px rgba(0,0,0,0.8)" }}
            >
              解析結果
            </h1>

            {loading && (
              <p className="text-white/90 mb-4">AIが栄養素をスキャン中...</p>
            )}

            {error && (
              <>
                <p className="text-red-400 mb-4">{error}</p>
                <button
                  onClick={handleReset}
                  className="w-full py-3 rounded-full font-bold bg-white/10 border border-white/20"
                >
                  ← 撮り直す
                </button>
              </>
            )}

            {!loading && !error && result && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div>
                  {imgPreview && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={imgPreview}
                      alt="解析した画像"
                      className="w-full rounded-2xl"
                    />
                  )}
                </div>

                <div>
                  {result.confidence < 10 ? (
                    <div className="bg-black/75 border-l-4 border-[#ff6b6b] rounded-2xl p-5 mb-4">
                      <b>AI判定:</b>{" "}
                      <span className="text-lg text-[#ff6b6b]">不明</span>
                      <br />
                      <small>確信度が低いため判定できませんでした</small>
                    </div>
                  ) : (
                    <div className="bg-black/75 border-l-4 border-[#00ff88] rounded-2xl p-5 mb-4">
                      <b>AI判定:</b>{" "}
                      <span className="text-lg text-[#00ff88]">
                        {result.name}
                      </span>
                      <br />
                      <small>確信度: {result.confidence.toFixed(1)}%</small>
                      <hr className="border-white/20 my-2" />
                      登録名: {result.full_name}
                      <br />
                      目安量: {result.portion}
                    </div>
                  )}

                  <div className="bg-black/80 border border-white/15 rounded-2xl p-5 text-center font-bold text-2xl mb-4">
                    <span className="block text-base font-normal text-white/70">
                      推定エネルギー
                    </span>
                    {result.confidence < 10
                      ? "不明"
                      : `${result.calories} kcal`}
                  </div>

                  {result.top3.length > 1 && (
                    <div>
                      <small className="text-white/60 font-bold">
                        AIの予測候補 TOP3
                      </small>
                      {result.top3.map((c, i) => {
                        const color = i === 0 ? "#00ff88" : "#aaaaaa";
                        const label = i === 0 ? "第1候補" : `第${i + 1}候補`;
                        return (
                          <div
                            key={c.name + i}
                            className="bg-black/75 rounded-lg px-3 py-2 mt-2 flex justify-between"
                            style={{ borderLeft: `3px solid ${color}` }}
                          >
                            <span>
                              <span className="text-xs mr-2" style={{ color }}>
                                {label}
                              </span>
                              <b>{c.name}</b>
                            </span>
                            <span className="text-white/60">
                              {c.confidence.toFixed(1)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}

            {!loading && !error && result && (
              <button
                onClick={handleReset}
                className="w-full py-3 rounded-full font-bold bg-white/10 border border-white/20 mt-6"
              >
                ← 撮り直す
              </button>
            )}
          </>
        )}
      </div>
    </main>
  );
}
