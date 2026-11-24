import { apiUrl, baseOriginUrl } from "./basePath";
import { Card, CardHeader, CardContent } from "./components/ui/card";
import { Alert, AlertDescription } from "./components/ui/alert";
import { Badge } from "./components/ui/badge";
import {
  Field,
  FieldLabel,
  FieldDescription,
  FieldError,
} from "./components/ui/field";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { NativeSelect } from "./components/ui/native-select";
import { Switch } from "./components/ui/switch";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs";
import { Send, Square, Copy, Plus, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router";
import {
  ENDPOINTS,
  appendAssistantTurnToBody,
  applyMessagesToBody,
  applySseFrameToResult,
  drainSseBuffer,
  emptyParsedResult,
  extractMessagesFromBody,
  formatCurlCommand,
  parseJsonResponse,
  type EndpointSpec,
  type ParsedApiResult,
  type SseFrame,
  type TurnMessage,
} from "./playground";

interface PlaygroundProps {
  model: string | null;
  loraAdapters?: string[];
  maxContextTokens?: number | null;
  decisionMode?: boolean;
  imageMode?: boolean;
}

type RequestTab = "structured" | "json";
type ResponseTab = "parsed" | "raw" | "sse" | "curl";

interface CallMetrics {
  status: number | null;
  statusText: string;
  ttftMs: number | null;
  e2eMs: number | null;
}

const DEFAULT_MODEL = "default";
const MAX_SSE_INSPECTOR_FRAMES = 512;
const MAX_SSE_BUFFER_CHARS = 1_048_576;
const REQUEST_IDLE_TIMEOUT_MS = 120_000;
const NON_STREAM_TIMEOUT_MS = 600_000;
const MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024;
const MAX_TOTAL_IMAGE_DATA_URL_CHARS = 28 * 1024 * 1024;
const MAX_IMAGES_PER_REQUEST = 4;
const MAX_IMAGE_FILE_BYTES = 16 * 1024 * 1024;
const MAX_CANVAS_DIM = 2048;

function formatDisplayJson(val: unknown): string {
  const formatted = JSON.stringify(
    val,
    (key, v) => {
      if (key === "b64_json" && typeof v === "string" && v.length > 256) {
        return `${v.slice(0, 64)}… [${v.length} base64 chars]`;
      }
      return v;
    },
    2,
  );
  return formatted.length > MAX_SSE_BUFFER_CHARS
    ? `${formatted.slice(0, MAX_SSE_BUFFER_CHARS)}\n… [response truncated at 1 MiB]`
    : formatted;
}

function endpointsForModel(isImageModel: boolean): EndpointSpec[] {
  if (isImageModel) {
    return ENDPOINTS.filter(
      (e) => e.kind === "openai-images" || e.group === "System",
    );
  }
  return ENDPOINTS.filter((e) => e.kind !== "openai-images");
}

function resolveRoute(
  pathEndpointId: string | undefined,
  searchParams: URLSearchParams,
  decisionMode: boolean,
  model?: string | null,
  imageMode = false,
): {
  endpoint: EndpointSpec;
  presetId: string;
} {
  const requestedEndpoint = searchParams.get("endpoint") ?? pathEndpointId;
  const isImageModel =
    imageMode || Boolean(model?.toLowerCase().includes("qwen-image"));
  const available = endpointsForModel(isImageModel);
  const fallbackId = decisionMode
    ? "decisions"
    : isImageModel
      ? "images-generations"
      : "chat-completions";
  const endpoint =
    available.find((e) => e.id === requestedEndpoint) ??
    available.find((e) => e.id === fallbackId) ??
    available[0];
  const requestedPreset = searchParams.get("preset");
  const preset =
    endpoint.presets.find((p) => p.id === requestedPreset) ??
    endpoint.presets[0];
  return { endpoint, presetId: preset.id };
}

function readFileAsEngineDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    if (file.size > MAX_IMAGE_FILE_BYTES) {
      reject(new Error(`Image '${file.name}' exceeds 16 MiB limit.`));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.onload = () => {
      const result = typeof reader.result === "string" ? reader.result : "";
      if (
        file.type === "image/png" ||
        file.type === "image/jpeg" ||
        result.startsWith("data:image/png;") ||
        result.startsWith("data:image/jpeg;")
      ) {
        resolve(result);
        return;
      }
      // Transcode WebP/GIF/BMP to PNG data URL so warp34's native Rust PNG/JPEG decoder accepts it
      const img = new Image();
      img.onload = () => {
        const rawW = img.naturalWidth || img.width || 64;
        const rawH = img.naturalHeight || img.height || 64;
        const scale = Math.min(1, MAX_CANVAS_DIM / Math.max(rawW, rawH, 1));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(rawW * scale));
        canvas.height = Math.max(1, Math.round(rawH * scale));
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(result);
          return;
        }
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/png"));
      };
      img.onerror = () =>
        reject(new Error(`Failed to decode image '${file.name}'.`));
      img.src = result;
    };
    reader.readAsDataURL(file);
  });
}

export function Playground({
  model,
  loraAdapters = [],
  maxContextTokens = null,
  decisionMode = false,
  imageMode = false,
}: PlaygroundProps) {
  const navigate = useNavigate();
  const { endpointId: pathEndpointId } = useParams<{ endpointId?: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const isImageModel =
    imageMode || Boolean(model?.toLowerCase().includes("qwen-image"));
  const availableEndpoints = useMemo(
    () => endpointsForModel(isImageModel),
    [isImageModel],
  );
  const defaultEndpointId = decisionMode
    ? "decisions"
    : isImageModel
      ? "images-generations"
      : "chat-completions";
  const initial = resolveRoute(
    pathEndpointId,
    searchParams,
    decisionMode,
    model,
    imageMode,
  );
  const [endpointId, setEndpointId] = useState<string>(initial.endpoint.id);
  const endpoint =
    availableEndpoints.find((e) => e.id === endpointId) ??
    availableEndpoints[0];

  const [presetId, setPresetId] = useState<string>(initial.presetId);
  const [selectedAdapter, setSelectedAdapter] = useState<string>("");
  const baseModel = model ?? DEFAULT_MODEL;
  const activeModel =
    selectedAdapter && loraAdapters.includes(selectedAdapter)
      ? selectedAdapter
      : baseModel;
  const maxTokensLimit =
    maxContextTokens && maxContextTokens > 0 ? maxContextTokens : 1_048_576;

  function updateRouteParams(nextEndpointId: string, nextPresetId: string) {
    const spec = ENDPOINTS.find((e) => e.id === nextEndpointId) ?? ENDPOINTS[0];
    const nextParams = new URLSearchParams();
    if (nextEndpointId !== defaultEndpointId) {
      nextParams.set("endpoint", nextEndpointId);
    }
    if (nextPresetId !== spec.presets[0].id) {
      nextParams.set("preset", nextPresetId);
    }
    if (pathEndpointId) {
      const qs = nextParams.toString();
      navigate(qs ? `/playground?${qs}` : "/playground", { replace: true });
    } else {
      setSearchParams(nextParams, { replace: true });
    }
  }

  const [rawJson, setRawJson] = useState<string>(() => {
    const preset =
      initial.endpoint.presets.find((p) => p.id === initial.presetId) ??
      initial.endpoint.presets[0];
    return initial.endpoint.method === "GET"
      ? ""
      : JSON.stringify(preset.buildBody(activeModel), null, 2);
  });
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string>("");
  const [requestTab, setRequestTab] = useState<RequestTab>(() =>
    initial.endpoint.conversational || initial.endpoint.kind === "openai-images"
      ? "structured"
      : "json",
  );
  const [responseTab, setResponseTab] = useState<ResponseTab>("parsed");

  const [inFlight, setInFlight] = useState<boolean>(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const [callMetrics, setCallMetrics] = useState<CallMetrics>({
    status: null,
    statusText: "",
    ttftMs: null,
    e2eMs: null,
  });
  const [parsedResult, setParsedResult] =
    useState<ParsedApiResult>(emptyParsedResult);
  const [rawResponse, setRawResponse] = useState<string>("");
  const [sseFrames, setSseFrames] = useState<SseFrame[]>([]);
  const [followUpText, setFollowUpText] = useState<string>("");
  const [copiedCurl, setCopiedCurl] = useState<boolean>(false);
  const [copyError, setCopyError] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const prevModelRef = useRef<string>(activeModel);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
    };
  }, []);

  // Keep the request JSON's `model` field synced when the first live metrics snapshot arrives
  useEffect(() => {
    if (inFlight || prevModelRef.current === activeModel) return;
    const prev = prevModelRef.current;
    prevModelRef.current = activeModel;
    try {
      const parsed = JSON.parse(rawJson) as Record<string, unknown>;
      if (parsed.model === prev || parsed.model === DEFAULT_MODEL) {
        parsed.model = activeModel;
        setRawJson(JSON.stringify(parsed, null, 2));
      }
    } catch {
      // Leave user's unparseable JSON untouched
    }
  }, [activeModel, rawJson, inFlight]);

  // Synchronize endpoint & preset state when URL path or search params change externally
  useEffect(() => {
    if (inFlight) return;
    const resolved = resolveRoute(
      pathEndpointId,
      searchParams,
      decisionMode,
      model,
      imageMode,
    );
    if (resolved.endpoint.id !== endpointId || resolved.presetId !== presetId) {
      setEndpointId(resolved.endpoint.id);
      setPresetId(resolved.presetId);
      const nextPreset =
        resolved.endpoint.presets.find((p) => p.id === resolved.presetId) ??
        resolved.endpoint.presets[0];
      setRawJson(
        resolved.endpoint.method === "GET"
          ? ""
          : JSON.stringify(nextPreset.buildBody(activeModel), null, 2),
      );
      setJsonError(null);
      setRequestTab(
        resolved.endpoint.conversational ||
          resolved.endpoint.kind === "openai-images"
          ? "structured"
          : "json",
      );
    }
  }, [pathEndpointId, searchParams, decisionMode, model, imageMode, inFlight]);

  function selectEndpoint(nextId: string) {
    if (inFlight) return;
    const nextSpec =
      availableEndpoints.find((e) => e.id === nextId) ?? availableEndpoints[0];
    const firstPreset = nextSpec.presets[0];
    setEndpointId(nextSpec.id);
    setPresetId(firstPreset.id);
    updateRouteParams(nextSpec.id, firstPreset.id);
    const body = firstPreset.buildBody(activeModel);
    setRawJson(nextSpec.method === "GET" ? "" : JSON.stringify(body, null, 2));
    setJsonError(null);
    setRequestTab(
      nextSpec.conversational || nextSpec.kind === "openai-images"
        ? "structured"
        : "json",
    );
    clearResponseState();
  }

  function selectPreset(nextPresetId: string) {
    if (inFlight) return;
    const preset =
      endpoint.presets.find((p) => p.id === nextPresetId) ??
      endpoint.presets[0];
    setPresetId(preset.id);
    updateRouteParams(endpoint.id, preset.id);
    const body = preset.buildBody(activeModel);
    setRawJson(endpoint.method === "GET" ? "" : JSON.stringify(body, null, 2));
    setJsonError(null);
    clearResponseState();
  }

  function clearResponseState() {
    setRequestError(null);
    setCallMetrics({ status: null, statusText: "", ttftMs: null, e2eMs: null });
    setParsedResult(emptyParsedResult());
    setRawResponse("");
    setSseFrames([]);
  }

  const bodyObj = useMemo<Record<string, unknown> | null>(() => {
    if (endpoint.method === "GET") return {};
    try {
      const val = JSON.parse(rawJson);
      if (!val || typeof val !== "object" || Array.isArray(val)) {
        return null;
      }
      return val as Record<string, unknown>;
    } catch {
      return null;
    }
  }, [endpoint.method, rawJson]);

  function updateBodyObject(
    updater: (body: Record<string, unknown>) => Record<string, unknown>,
  ) {
    if (!bodyObj) return;
    const updated = updater({ ...bodyObj });
    setRawJson(JSON.stringify(updated, null, 2));
    setJsonError(null);
  }
  const messages: TurnMessage[] =
    endpoint.conversational && bodyObj
      ? extractMessagesFromBody(endpoint.kind, bodyObj)
      : [];

  const totalImages = messages.reduce(
    (sum, m) => sum + (m.images?.length ?? 0),
    0,
  );

  function handleMessageChange(index: number, patch: Partial<TurnMessage>) {
    updateBodyObject((curr) => {
      const list = extractMessagesFromBody(endpoint.kind, curr).map((m, i) =>
        i === index ? { ...m, ...patch } : m,
      );
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  async function handleAttachImages(index: number, files: FileList | File[]) {
    if (inFlight) return;
    const fileArr = Array.from(files).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (fileArr.length === 0) return;
    const remainingBudget = Math.max(0, MAX_IMAGES_PER_REQUEST - totalImages);
    const toRead = fileArr.slice(0, remainingBudget);
    if (toRead.length === 0) return;
    try {
      const requestAtStart = abortRef.current;
      const dataUrls = await Promise.all(toRead.map(readFileAsEngineDataUrl));
      // A request started while decoding: do not mutate its editor snapshot.
      if (abortRef.current !== requestAtStart) return;
      const existingChars = messages.reduce(
        (sum, m) =>
          sum + (m.images?.reduce((acc, u) => acc + u.length, 0) ?? 0),
        0,
      );
      const addedChars = dataUrls.reduce((acc, u) => acc + u.length, 0);
      if (existingChars + addedChars > MAX_TOTAL_IMAGE_DATA_URL_CHARS) {
        setRequestError(
          "Attached images exceed the 32 MiB request body limit.",
        );
        return;
      }
      setRequestError(null);
      updateBodyObject((curr) => {
        const list = extractMessagesFromBody(endpoint.kind, curr).map((m, i) =>
          i === index
            ? { ...m, images: [...(m.images ?? []), ...dataUrls] }
            : m,
        );
        return applyMessagesToBody(endpoint.kind, curr, list);
      });
    } catch (err) {
      setRequestError(
        err instanceof Error ? err.message : "Failed to attach image.",
      );
    }
  }

  function handleRemoveImage(turnIdx: number, imgIdx: number) {
    updateBodyObject((curr) => {
      const list = extractMessagesFromBody(endpoint.kind, curr).map((m, i) => {
        if (i !== turnIdx || !m.images) return m;
        const nextImgs = m.images.filter((_, j) => j !== imgIdx);
        return { ...m, images: nextImgs.length > 0 ? nextImgs : undefined };
      });
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  function handleAddMessage(role: TurnMessage["role"] = "user") {
    updateBodyObject((curr) => {
      const list = [
        ...extractMessagesFromBody(endpoint.kind, curr),
        { role, content: "" },
      ];
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  function handleRemoveMessage(index: number) {
    updateBodyObject((curr) => {
      const list = extractMessagesFromBody(endpoint.kind, curr).filter(
        (_, i) => i !== index,
      );
      return applyMessagesToBody(endpoint.kind, curr, list);
    });
  }

  function handleSendGeneratedToImageEdit(imageUrl: string) {
    if (inFlight) return;
    const editSpec = ENDPOINTS.find((e) => e.id === "images-edits");
    if (!editSpec) return;
    const preset = editSpec.presets[0];
    setEndpointId(editSpec.id);
    setPresetId(preset.id);
    updateRouteParams(editSpec.id, preset.id);
    const baseBody = preset.buildBody(activeModel);
    const nextBody: Record<string, unknown> = {
      ...baseBody,
      image: [imageUrl],
    };
    setRawJson(JSON.stringify(nextBody, null, 2));
    setJsonError(null);
    setRequestTab("structured");
  }

  async function handleAttachConditionImages(files: FileList | File[]) {
    const fileArr = Array.from(files).filter((f) =>
      f.type.startsWith("image/"),
    );
    if (fileArr.length === 0) return;
    const currentImgs = extractConditionImages(bodyObj);
    const remainingBudget = Math.max(
      0,
      MAX_IMAGES_PER_REQUEST - currentImgs.length,
    );
    const toRead = fileArr.slice(0, remainingBudget);
    if (toRead.length === 0) return;
    try {
      const dataUrls = await Promise.all(toRead.map(readFileAsEngineDataUrl));
      setRequestError(null);
      updateBodyObject((curr) => {
        const nextImgs = [...extractConditionImages(curr), ...dataUrls];
        const next: Record<string, unknown> = { ...curr, image: nextImgs };
        delete next.images;
        return next;
      });
    } catch (err) {
      setRequestError(
        err instanceof Error ? err.message : "Failed to attach image.",
      );
    }
  }

  function handleRemoveConditionImage(imgIdx: number) {
    updateBodyObject((curr) => {
      const nextImgs = extractConditionImages(curr).filter(
        (_, i) => i !== imgIdx,
      );
      const next = { ...curr };
      delete next.images;
      if (nextImgs.length > 0) {
        next.image = nextImgs;
      } else {
        delete next.image;
      }
      return next;
    });
  }

  function handleRawJsonEdit(value: string) {
    setRawJson(value);
    if (endpoint.method === "GET") {
      setJsonError(null);
      return;
    }
    try {
      const parsed = JSON.parse(value);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        setJsonError("Request body must be a JSON object");
      } else {
        setJsonError(null);
      }
    } catch (err) {
      setJsonError((err as Error).message);
    }
  }

  async function executeRequest(overrideJson?: string) {
    if (inFlight) return;
    const payloadText = overrideJson ?? rawJson;
    let reqObj: Record<string, unknown> | null = null;
    if (endpoint.method !== "GET") {
      if (
        new TextEncoder().encode(payloadText).byteLength >
        MAX_REQUEST_BODY_BYTES
      ) {
        setRequestError(
          "Request body exceeds 32 MiB server limit (reduce attached image size or count).",
        );
        return;
      }
      try {
        reqObj = JSON.parse(payloadText) as Record<string, unknown>;
      } catch (err) {
        setJsonError((err as Error).message);
        return;
      }
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    let timedOut = false;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    const initialTimeoutMs =
      endpoint.method !== "GET" && reqObj?.stream !== true
        ? NON_STREAM_TIMEOUT_MS
        : REQUEST_IDLE_TIMEOUT_MS;
    const resetIdleTimer = (ms: number = REQUEST_IDLE_TIMEOUT_MS) => {
      if (idleTimer !== null) clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        timedOut = true;
        controller.abort();
      }, ms);
    };
    const clearIdleTimer = () => {
      if (idleTimer !== null) {
        clearTimeout(idleTimer);
        idleTimer = null;
      }
    };

    setInFlight(true);
    setRequestError(null);
    setCallMetrics({ status: null, statusText: "", ttftMs: null, e2eMs: null });
    setParsedResult(emptyParsedResult());
    setRawResponse("");
    setSseFrames([]);

    const headers: Record<string, string> = {};
    if (endpoint.method !== "GET") {
      headers["Content-Type"] = "application/json";
    }
    const needsStatefulScope =
      endpoint.kind === "openai-responses" &&
      reqObj !== null &&
      (reqObj.store === true ||
        typeof reqObj.previous_response_id === "string");
    const effectiveKey =
      apiKey.trim().length > 0
        ? apiKey.trim()
        : needsStatefulScope
          ? "playground-local"
          : "";
    if (effectiveKey.length > 0) {
      if (endpoint.kind === "anthropic-messages") {
        headers["x-api-key"] = effectiveKey;
      } else {
        headers["Authorization"] = `Bearer ${effectiveKey}`;
      }
    }

    const startTime = performance.now();
    let firstByteMs: number | null = null;

    try {
      resetIdleTimer(initialTimeoutMs);
      const response = await fetch(apiUrl(endpoint.path), {
        method: endpoint.method,
        headers,
        body: endpoint.method === "GET" ? undefined : payloadText,
        signal: controller.signal,
      });
      resetIdleTimer();

      const contentType = response.headers.get("content-type") ?? "";
      const isEventStream = contentType.includes("text/event-stream");

      if (isEventStream && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let frameIndex = 0;
        let runningParsed = emptyParsedResult();
        const retainedFrames: SseFrame[] = [];

        while (true) {
          resetIdleTimer();
          const { value, done } = await reader.read();
          if (done) break;
          const nowMs = Math.round(performance.now() - startTime);
          if (firstByteMs === null) {
            firstByteMs = nowMs;
            setCallMetrics({
              status: response.status,
              statusText: response.statusText || "OK",
              ttftMs: firstByteMs,
              e2eMs: null,
            });
          }
          buffer += decoder.decode(value, { stream: true });
          if (buffer.length > MAX_SSE_BUFFER_CHARS) {
            throw new Error("SSE frame buffer exceeded 1 MiB limit.");
          }
          const drained = drainSseBuffer(buffer, frameIndex, nowMs);
          buffer = drained.remainder;
          if (drained.frames.length > 0) {
            frameIndex += drained.frames.length;
            for (const f of drained.frames) {
              runningParsed = applySseFrameToResult(
                endpoint.kind,
                runningParsed,
                f,
              );
              if (retainedFrames.length < MAX_SSE_INSPECTOR_FRAMES) {
                retainedFrames.push(f);
              }
            }
            setSseFrames(retainedFrames.slice());
            setParsedResult(runningParsed);
            if (runningParsed.streamError) {
              setRequestError(runningParsed.streamError);
            }
            const lastData = drained.frames.at(-1);
            if (lastData && lastData.data !== "[DONE]") {
              setRawResponse(lastData.data);
            }
          }
        }

        const totalMs = Math.round(performance.now() - startTime);
        setCallMetrics({
          status: response.status,
          statusText: response.statusText || "OK",
          ttftMs: firstByteMs ?? totalMs,
          e2eMs: totalMs,
        });
      } else {
        const rawText = await response.text();
        const totalMs = Math.round(performance.now() - startTime);
        setCallMetrics({
          status: response.status,
          statusText: response.statusText || (response.ok ? "OK" : "Error"),
          ttftMs: totalMs,
          e2eMs: totalMs,
        });
        try {
          const jsonVal = JSON.parse(rawText);
          const displayJson = formatDisplayJson(jsonVal);
          setRawResponse(displayJson);
          if (response.ok) {
            setParsedResult(parseJsonResponse(endpoint.kind, jsonVal));
          } else {
            setParsedResult({
              ...emptyParsedResult(),
              content: displayJson,
            });
          }
        } catch {
          const text =
            rawText.length > MAX_SSE_BUFFER_CHARS
              ? `${rawText.slice(0, MAX_SSE_BUFFER_CHARS)}\n… [response truncated at 1 MiB]`
              : rawText;
          setRawResponse(text);
          setParsedResult({
            ...emptyParsedResult(),
            content: text,
          });
        }
      }
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setRequestError(
          timedOut
            ? "Request timed out waiting for server activity."
            : "Request aborted by user.",
        );
      } else {
        setRequestError((err as Error).message || "Network request failed.");
      }
    } finally {
      clearIdleTimer();
      setInFlight(false);
    }
  }

  function handleAppendAndContinue(sendImmediately: boolean) {
    if (!bodyObj) return;
    const nextBody = appendAssistantTurnToBody(
      endpoint.kind,
      bodyObj,
      parsedResult,
      followUpText,
    );
    const nextJson = JSON.stringify(nextBody, null, 2);
    setRawJson(nextJson);
    setJsonError(null);
    setFollowUpText("");
    if (
      sendImmediately &&
      (followUpText.trim().length > 0 || parsedResult.toolCalls.length > 0)
    ) {
      void executeRequest(nextJson);
    }
  }

  const curlCommand = formatCurlCommand(
    baseOriginUrl(),
    endpoint,
    rawJson,
    apiKey,
  );

  async function handleCopyCurl() {
    setCopyError(false);
    try {
      if (!navigator.clipboard?.writeText)
        throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(curlCommand);
      setCopiedCurl(true);
      if (copyTimerRef.current !== null) clearTimeout(copyTimerRef.current);
      copyTimerRef.current = setTimeout(() => setCopiedCurl(false), 1500);
    } catch {
      setCopiedCurl(false);
      setCopyError(true);
      setResponseTab("curl");
    }
  }

  const decodeTokPerSec =
    parsedResult.usage.outputTokens &&
    parsedResult.usage.outputTokens > 1 &&
    callMetrics.e2eMs !== null &&
    callMetrics.ttftMs !== null &&
    callMetrics.e2eMs > callMetrics.ttftMs
      ? (
          (parsedResult.usage.outputTokens * 1000) /
          (callMetrics.e2eMs - callMetrics.ttftMs)
        ).toFixed(1)
      : null;

  const hasAssistantReply =
    endpoint.conversational &&
    (parsedResult.content.length > 0 || parsedResult.toolCalls.length > 0);

  return (
    <div className="playground">
      <div className="section-heading">
        <div>
          <h1>API Playground</h1>
          <p>
            Inspect wire payloads, SSE frames, prefix cache hits, and multi-turn
            state against this engine.
          </p>
        </div>
        <Field className="playground-auth w-auto gap-2">
          <FieldLabel htmlFor="pg-api-key" className="eyebrow text-[11px]">
            API KEY
          </FieldLabel>
          <Input
            disabled={inFlight}
            id="pg-api-key"
            type="password"
            placeholder="Optional Bearer / x-api-key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoComplete="off"
          />
        </Field>
      </div>

      <div className="playground-workspace">
        <Card
          className="pg-sidebar gap-5 p-4 max-[1051px]:hidden"
          role="complementary"
          aria-label="API endpoints"
        >
          {(
            [
              "Generative",
              "Decision & Scoring",
              "Tokenizer & Counts",
              "System",
            ] as const
          ).map((group) => {
            const groupEps = availableEndpoints.filter(
              (ep) => ep.group === group,
            );
            if (groupEps.length === 0) return null;
            return (
              <div key={group} className="pg-sidebar-group">
                <span className="eyebrow text-[11px]">
                  {group.toUpperCase()}
                </span>
                <div className="pg-sidebar-items">
                  {groupEps.map((ep) => (
                    <Button
                      key={ep.id}
                      variant="ghost"
                      disabled={inFlight}
                      type="button"
                      aria-current={endpoint.id === ep.id ? "true" : undefined}
                      title={ep.path}
                      className="w-full justify-start border border-transparent px-2 text-left text-muted-foreground aria-current:border-input aria-current:bg-accent aria-current:text-foreground"
                      onClick={() => selectEndpoint(ep.id)}
                    >
                      <MethodBadge method={ep.method} />
                      <span className="pg-sidebar-path mono">{ep.path}</span>
                    </Button>
                  ))}
                </div>
              </div>
            );
          })}
        </Card>

        <div className="pg-main">
          <Card
            role="region"
            className="playground-toolbar flex-row gap-3 p-4"
            aria-label="Endpoint controls"
          >
            <Field className="pg-field gap-2 min-[1051px]:hidden">
              <FieldLabel htmlFor="pg-endpoint" className="eyebrow text-[11px]">
                ENDPOINT
              </FieldLabel>
              <NativeSelect
                disabled={inFlight}
                id="pg-endpoint"
                value={endpoint.id}
                onChange={(e) => selectEndpoint(e.target.value)}
              >
                {(
                  [
                    "Generative",
                    "Decision & Scoring",
                    "Tokenizer & Counts",
                    "System",
                  ] as const
                ).map((group) => {
                  const groupEps = availableEndpoints.filter(
                    (ep) => ep.group === group,
                  );
                  if (groupEps.length === 0) return null;
                  return (
                    <optgroup key={group} label={group}>
                      {groupEps.map((ep) => (
                        <option key={ep.id} value={ep.id}>
                          {ep.label}
                        </option>
                      ))}
                    </optgroup>
                  );
                })}
              </NativeSelect>
            </Field>

            <Field className="pg-field gap-2">
              <FieldLabel htmlFor="pg-preset" className="eyebrow text-[11px]">
                PRESET
              </FieldLabel>
              <NativeSelect
                disabled={inFlight}
                id="pg-preset"
                aria-describedby="pg-preset-hint"
                value={presetId}
                onChange={(e) => selectPreset(e.target.value)}
              >
                {endpoint.presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </NativeSelect>
              <FieldDescription id="pg-preset-hint">
                {endpoint.presets.find((p) => p.id === presetId)?.description}
              </FieldDescription>
            </Field>

            {loraAdapters.length > 0 && endpoint.method !== "GET" && (
              <Field className="pg-field gap-2">
                <FieldLabel htmlFor="pg-model" className="eyebrow text-[11px]">
                  MODEL / LORA
                </FieldLabel>
                <NativeSelect
                  disabled={inFlight}
                  id="pg-model"
                  value={
                    selectedAdapter && loraAdapters.includes(selectedAdapter)
                      ? selectedAdapter
                      : ""
                  }
                  onChange={(e) => setSelectedAdapter(e.target.value)}
                >
                  <option value="">{baseModel} (base)</option>
                  {loraAdapters.map((adapter) => (
                    <option key={adapter} value={adapter}>
                      {adapter} (LoRA)
                    </option>
                  ))}
                </NativeSelect>
              </Field>
            )}

            <div className="pg-actions">
              <Button
                type="button"
                variant="outline"
                onClick={() => void handleCopyCurl()}
              >
                <Copy aria-hidden="true" />
                <span aria-live="polite">
                  {copiedCurl ? "Copied cURL" : "Copy cURL"}
                </span>
              </Button>
              <Button
                type="button"
                variant={inFlight ? "destructive" : "default"}
                disabled={!inFlight && Boolean(jsonError)}
                onClick={() =>
                  inFlight ? abortRef.current?.abort() : void executeRequest()
                }
              >
                {inFlight ? (
                  <Square aria-hidden="true" />
                ) : (
                  <Send aria-hidden="true" />
                )}
                {inFlight ? "Abort" : "Send Request"}
              </Button>
            </div>
          </Card>

          {copyError && (
            <Alert variant="destructive">
              <AlertDescription>
                Could not copy. Select and copy the command from the cURL tab.
              </AlertDescription>
            </Alert>
          )}
          <div className="playground-grid">
            <Card
              role="region"
              className="playground-pane gap-0 p-4"
              aria-labelledby="req-pane-title"
            >
              <Tabs
                value={requestTab}
                onValueChange={(value) => {
                  if (value === "structured" || value === "json")
                    setRequestTab(value);
                }}
              >
                <CardHeader className="pg-pane-heading flex flex-row flex-wrap items-center justify-between px-0">
                  <div className="pg-title-group">
                    <MethodBadge method={endpoint.method} />
                    <h2 id="req-pane-title" className="mono">
                      {endpoint.path}
                    </h2>
                  </div>
                  {(endpoint.conversational ||
                    endpoint.kind === "openai-images") && (
                    <TabsList aria-label="Request editor mode">
                      <TabsTrigger value="structured">
                        {endpoint.kind === "openai-images"
                          ? "Prompt & Params"
                          : "Messages & Params"}
                      </TabsTrigger>
                      <TabsTrigger value="json">JSON Body</TabsTrigger>
                    </TabsList>
                  )}
                </CardHeader>

                <CardContent className="px-0">
                  <fieldset
                    disabled={inFlight}
                    className="min-w-0 border-0 p-0 m-0"
                  >
                    <TabsContent value="structured">
                      {endpoint.kind === "openai-images" && bodyObj ? (
                        <ImageRequestEditor
                          endpoint={endpoint}
                          body={bodyObj}
                          onUpdate={updateBodyObject}
                          onAttachConditionImages={handleAttachConditionImages}
                          onRemoveConditionImage={handleRemoveConditionImage}
                          onSubmit={() => void executeRequest()}
                          disabled={inFlight}
                        />
                      ) : endpoint.conversational && bodyObj ? (
                        <div className="pg-structured">
                          <ParameterBar
                            endpoint={endpoint}
                            body={bodyObj}
                            maxTokensLimit={maxTokensLimit}
                            onUpdate={updateBodyObject}
                            disabled={inFlight}
                          />

                          <div className="pg-messages-header">
                            <span className="eyebrow text-[11px]">
                              {endpoint.kind === "openai-responses"
                                ? "INSTRUCTIONS & INPUT TURNS"
                                : "MESSAGES ARRAY"}
                            </span>
                            <div className="pg-message-add-btns">
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => handleAddMessage("user")}
                              >
                                <Plus aria-hidden="true" /> User turn
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => handleAddMessage("assistant")}
                              >
                                <Plus aria-hidden="true" /> Assistant turn
                              </Button>
                            </div>
                          </div>

                          <div className="pg-messages-list">
                            {messages.map((msg, idx) => (
                              <div
                                key={idx}
                                className="pg-message-row"
                                onDragOver={(e) => {
                                  if (msg.role === "user") {
                                    e.preventDefault();
                                  }
                                }}
                                onDrop={(e) => {
                                  if (
                                    msg.role === "user" &&
                                    e.dataTransfer?.files?.length
                                  ) {
                                    e.preventDefault();
                                    void handleAttachImages(
                                      idx,
                                      e.dataTransfer.files,
                                    );
                                  }
                                }}
                              >
                                <div className="pg-message-meta">
                                  <NativeSelect
                                    aria-label={`Turn ${idx + 1} role`}
                                    value={msg.role}
                                    onChange={(e) => {
                                      const role = e.target.value;
                                      if (
                                        role === "system" ||
                                        role === "user" ||
                                        role === "assistant" ||
                                        role === "tool"
                                      ) {
                                        handleMessageChange(idx, { role });
                                      }
                                    }}
                                  >
                                    <option value="system">system</option>
                                    <option value="user">user</option>
                                    <option value="assistant">assistant</option>
                                    {endpoint.kind === "openai-chat" && (
                                      <option value="tool">tool</option>
                                    )}
                                  </NativeSelect>
                                  <span className="pg-turn-index mono">
                                    #{idx}
                                  </span>
                                  {msg.tool_calls &&
                                    msg.tool_calls.length > 0 && (
                                      <Badge
                                        variant="secondary"
                                        className="text-[var(--accent-bright)]"
                                      >
                                        {msg.tool_calls.length} tool_call
                                        {msg.tool_calls.length > 1 ? "s" : ""}
                                      </Badge>
                                    )}
                                  <div className="pg-turn-actions">
                                    {msg.role === "user" &&
                                      totalImages < MAX_IMAGES_PER_REQUEST && (
                                        <Button
                                          asChild
                                          variant="outline"
                                          size="sm"
                                          className="focus-within:ring-2 focus-within:ring-ring"
                                        >
                                          <label>
                                            <Plus aria-hidden="true" /> Image
                                            <input
                                              type="file"
                                              accept="image/png,image/jpeg,image/webp,image/gif"
                                              multiple
                                              aria-label={`Attach image to turn ${idx + 1}`}
                                              className="sr-only"
                                              onChange={(e) => {
                                                if (
                                                  e.target.files &&
                                                  e.target.files.length > 0
                                                ) {
                                                  void handleAttachImages(
                                                    idx,
                                                    e.target.files,
                                                  );
                                                  e.target.value = "";
                                                }
                                              }}
                                            />
                                          </label>
                                        </Button>
                                      )}
                                    <Button
                                      type="button"
                                      variant="ghost"
                                      size="icon-xs"
                                      aria-label={`Remove turn ${idx + 1}`}
                                      onClick={() => handleRemoveMessage(idx)}
                                    >
                                      <X aria-hidden="true" />
                                    </Button>
                                  </div>
                                </div>

                                {msg.images && msg.images.length > 0 && (
                                  <div className="pg-turn-images">
                                    {msg.images.map((imgUrl, imgIdx) => (
                                      <div
                                        key={imgIdx}
                                        className="pg-turn-thumb"
                                      >
                                        <img
                                          src={imgUrl}
                                          alt={`Turn ${idx + 1} attachment ${imgIdx + 1}`}
                                          className="pg-turn-thumb-img"
                                        />
                                        <span className="pg-turn-thumb-meta mono">
                                          {imgUrl.startsWith("data:image/jpeg")
                                            ? "JPEG"
                                            : imgUrl.startsWith(
                                                  "data:image/png",
                                                )
                                              ? "PNG"
                                              : "IMG"}{" "}
                                          #{imgIdx + 1}
                                        </span>
                                        <Button
                                          type="button"
                                          variant="ghost"
                                          size="icon-xs"
                                          className="pg-thumb-remove"
                                          aria-label={`Remove image ${imgIdx + 1} from turn ${idx + 1}`}
                                          onClick={() =>
                                            handleRemoveImage(idx, imgIdx)
                                          }
                                        >
                                          <X aria-hidden="true" />
                                        </Button>
                                      </div>
                                    ))}
                                  </div>
                                )}

                                <Textarea
                                  aria-label={`Turn ${idx + 1} content`}
                                  rows={Math.min(
                                    8,
                                    Math.max(2, msg.content.split("\n").length),
                                  )}
                                  value={msg.content}
                                  placeholder={
                                    msg.role === "user"
                                      ? "user content (or drop / paste an image)…"
                                      : `${msg.role} content…`
                                  }
                                  onChange={(e) =>
                                    handleMessageChange(idx, {
                                      content: e.target.value,
                                    })
                                  }
                                  onPaste={(e) => {
                                    if (
                                      msg.role === "user" &&
                                      e.clipboardData?.files?.length
                                    ) {
                                      const imgFiles = Array.from(
                                        e.clipboardData.files,
                                      ).filter((f) =>
                                        f.type.startsWith("image/"),
                                      );
                                      if (imgFiles.length > 0) {
                                        e.preventDefault();
                                        void handleAttachImages(idx, imgFiles);
                                      }
                                    }
                                  }}
                                  onKeyDown={(e) => {
                                    if (
                                      (e.metaKey || e.ctrlKey) &&
                                      e.key === "Enter"
                                    ) {
                                      e.preventDefault();
                                      void executeRequest();
                                    }
                                  }}
                                />
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <p className="muted">
                          Correct the request in JSON Body to use the structured
                          editor.
                        </p>
                      )}
                    </TabsContent>
                    <TabsContent
                      value="json"
                      {...(!endpoint.conversational &&
                      endpoint.kind !== "openai-images"
                        ? {
                            role: "region",
                            "aria-labelledby": "req-pane-title",
                          }
                        : {})}
                    >
                      {endpoint.method === "GET" ? (
                        <div className="pg-empty-body">
                          <code>GET {endpoint.path}</code> takes no request
                          body. Click <b>Send Request</b> to inspect the
                          response.
                        </div>
                      ) : (
                        <Field
                          className="pg-json-editor gap-2"
                          data-invalid={Boolean(jsonError)}
                        >
                          <FieldLabel
                            htmlFor="pg-json-body"
                            className="sr-only"
                          >
                            Request JSON body
                          </FieldLabel>
                          <Textarea
                            id="pg-json-body"
                            aria-invalid={Boolean(jsonError)}
                            aria-describedby={
                              jsonError
                                ? "pg-json-hint pg-json-error"
                                : "pg-json-hint"
                            }
                            className="mono"
                            rows={18}
                            value={rawJson}
                            onChange={(e) => handleRawJsonEdit(e.target.value)}
                            onKeyDown={(e) => {
                              if (
                                (e.metaKey || e.ctrlKey) &&
                                e.key === "Enter"
                              ) {
                                e.preventDefault();
                                void executeRequest();
                              }
                            }}
                          />
                          <FieldDescription id="pg-json-hint">
                            Enter a JSON object. Press Ctrl or Command + Enter
                            to send.
                          </FieldDescription>
                          {jsonError && (
                            <FieldError id="pg-json-error">
                              JSON syntax error: {jsonError}
                            </FieldError>
                          )}
                        </Field>
                      )}
                    </TabsContent>
                  </fieldset>
                </CardContent>
              </Tabs>
            </Card>

            <Card
              role="region"
              className="playground-pane gap-0 p-4"
              aria-labelledby="resp-pane-title"
            >
              <Tabs
                value={responseTab}
                onValueChange={(value) => {
                  if (
                    value === "parsed" ||
                    value === "raw" ||
                    value === "sse" ||
                    value === "curl"
                  )
                    setResponseTab(value);
                }}
              >
                <CardHeader className="pg-pane-heading flex flex-row flex-wrap items-center justify-between px-0">
                  <div className="pg-title-group">
                    <h2 id="resp-pane-title">Wire Response</h2>
                    {inFlight && (
                      <Badge
                        variant="secondary"
                        className="text-[var(--accent-bright)]"
                      >
                        Streaming…
                      </Badge>
                    )}
                    {callMetrics.status !== null && (
                      <Badge
                        variant={
                          callMetrics.status >= 200 && callMetrics.status < 300
                            ? "secondary"
                            : "destructive"
                        }
                        className={
                          callMetrics.status >= 200 && callMetrics.status < 300
                            ? "text-[var(--aqua)]"
                            : undefined
                        }
                      >
                        {callMetrics.status} {callMetrics.statusText}
                      </Badge>
                    )}
                  </div>
                  <TabsList aria-label="Response inspector view">
                    <TabsTrigger value="parsed">Parsed</TabsTrigger>
                    <TabsTrigger value="raw">Raw JSON</TabsTrigger>
                    <TabsTrigger value="sse">
                      SSE ({sseFrames.length})
                    </TabsTrigger>
                    <TabsTrigger value="curl">cURL</TabsTrigger>
                  </TabsList>
                </CardHeader>

                <CardContent className="px-0">
                  <div
                    className="pg-metrics-strip"
                    aria-label="Response telemetry"
                  >
                    <div>
                      <span className="eyebrow text-[11px]">TTFT</span>
                      <strong className="numeric">
                        {callMetrics.ttftMs !== null
                          ? `${callMetrics.ttftMs} ms`
                          : "—"}
                      </strong>
                    </div>
                    <div>
                      <span className="eyebrow text-[11px]">TOTAL E2E</span>
                      <strong className="numeric">
                        {callMetrics.e2eMs !== null
                          ? `${callMetrics.e2eMs} ms`
                          : "—"}
                      </strong>
                    </div>
                    <div>
                      <span className="eyebrow text-[11px]">PACE</span>
                      <strong className="numeric">
                        {decodeTokPerSec ? `${decodeTokPerSec} tok/s` : "—"}
                      </strong>
                    </div>
                    <div>
                      <span className="eyebrow text-[11px]">IN / CACHED</span>
                      <strong className="numeric">
                        {parsedResult.usage.inputTokens ?? "—"} /{" "}
                        <span
                          className={
                            (parsedResult.usage.cachedTokens ?? 0) > 0
                              ? "healthy"
                              : ""
                          }
                        >
                          {parsedResult.usage.cachedTokens ?? "—"}
                        </span>
                      </strong>
                    </div>
                    <div>
                      <span className="eyebrow text-[11px]">OUTPUT</span>
                      <strong className="numeric">
                        {parsedResult.usage.outputTokens ?? "—"}
                      </strong>
                    </div>
                  </div>

                  {requestError && (
                    <Alert variant="destructive">
                      <AlertDescription>
                        <span>
                          <strong>Request failed:</strong> {requestError}
                        </span>
                      </AlertDescription>
                    </Alert>
                  )}

                  <TabsContent value="parsed">
                    <div className="pg-parsed-view">
                      {parsedResult.reasoning && (
                        <details className="pg-block pg-reasoning-block" open>
                          <summary className="eyebrow text-[11px]">
                            REASONING / THINKING STREAM
                            {parsedResult.usage.reasoningTokens !== null &&
                              ` (${parsedResult.usage.reasoningTokens} tok)`}
                          </summary>
                          <pre className="mono">{parsedResult.reasoning}</pre>
                        </details>
                      )}

                      {parsedResult.toolCalls.length > 0 && (
                        <div className="pg-block">
                          <span className="eyebrow text-[11px]">
                            TOOL CALLS
                          </span>
                          {parsedResult.toolCalls.map((tc) => (
                            <div key={tc.id} className="pg-tool-call mono">
                              <div className="pg-tool-header">
                                <strong>{tc.name}</strong>
                                <span className="muted">id: {tc.id}</span>
                              </div>
                              <pre>{tc.arguments}</pre>
                            </div>
                          ))}
                        </div>
                      )}

                      <div className="pg-block">
                        <div className="pg-block-header">
                          <span className="eyebrow text-[11px]">
                            {endpoint.conversational
                              ? "ASSISTANT OUTPUT"
                              : endpoint.kind === "openai-images"
                                ? "GENERATED IMAGE"
                                : "RESPONSE PAYLOAD"}
                          </span>
                          {parsedResult.finishReason && (
                            <span className="mono muted">
                              stop: {parsedResult.finishReason}
                            </span>
                          )}
                        </div>
                        {parsedResult.images &&
                          parsedResult.images.length > 0 && (
                            <div
                              className="pg-image-output-grid"
                              style={{
                                display: "grid",
                                gridTemplateColumns:
                                  "repeat(auto-fit, minmax(260px, 1fr))",
                                gap: "12px",
                                marginBottom: "12px",
                              }}
                            >
                              {parsedResult.images.map((src, idx) => (
                                <figure
                                  key={idx}
                                  style={{
                                    margin: 0,
                                    padding: "8px",
                                    borderRadius: "6px",
                                    border: "1px solid var(--border, #2a2f3a)",
                                    background:
                                      "repeating-conic-gradient(#1b1f27 0% 25%, #242936 0% 50%) 50% / 16px 16px",
                                  }}
                                >
                                  <img
                                    src={src}
                                    alt={`Generated output ${idx + 1}`}
                                    style={{
                                      width: "100%",
                                      height: "auto",
                                      display: "block",
                                      borderRadius: "4px",
                                      background:
                                        "repeating-conic-gradient(#161a22 0% 25%, #262d3d 0% 50%) 50% / 20px 20px",
                                    }}
                                  />
                                  <figcaption
                                    className="mono muted"
                                    style={{
                                      display: "flex",
                                      justifyContent: "space-between",
                                      alignItems: "center",
                                      gap: "8px",
                                      marginTop: "6px",
                                      fontSize: "12px",
                                    }}
                                  >
                                    <span>Output #{idx + 1}</span>
                                    <span
                                      style={{
                                        display: "inline-flex",
                                        alignItems: "center",
                                        gap: "8px",
                                      }}
                                    >
                                      <Button
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        disabled={inFlight}
                                        onClick={() =>
                                          handleSendGeneratedToImageEdit(src)
                                        }
                                      >
                                        ↪ Edit Image
                                      </Button>
                                      <a
                                        href={src}
                                        download={`qwen-image-${idx + 1}.png`}
                                        className="text-primary underline underline-offset-4"
                                      >
                                        Download PNG
                                      </a>
                                    </span>
                                  </figcaption>
                                </figure>
                              ))}
                            </div>
                          )}
                        <pre className="pg-output-pre mono">
                          {parsedResult.content ||
                            (inFlight
                              ? endpoint.kind === "openai-images"
                                ? "Synthesizing image…"
                                : "Waiting for tokens…"
                              : "Send a request to inspect the response.")}
                        </pre>
                      </div>

                      {hasAssistantReply && !inFlight && (
                        <div className="pg-continue-box">
                          <div className="pg-continue-header">
                            <span className="eyebrow text-[11px]">
                              MULTI-TURN CONTINUATION
                            </span>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => handleAppendAndContinue(false)}
                            >
                              <Plus aria-hidden="true" /> Append assistant turn
                            </Button>
                          </div>
                          <div className="pg-continue-row">
                            <Input
                              type="text"
                              aria-label="Follow-up user message"
                              placeholder={
                                parsedResult.toolCalls.length > 0
                                  ? 'Tool output JSON (default: {"status":"ok"})…'
                                  : "Type next user turn to append & send (tests prefix cache reuse)…"
                              }
                              value={followUpText}
                              onChange={(e) => setFollowUpText(e.target.value)}
                              onKeyDown={(e) => {
                                if (
                                  e.key === "Enter" &&
                                  (followUpText.trim().length > 0 ||
                                    parsedResult.toolCalls.length > 0)
                                ) {
                                  e.preventDefault();
                                  handleAppendAndContinue(true);
                                }
                              }}
                            />
                            <Button
                              type="button"
                              variant="default"
                              disabled={
                                followUpText.trim().length === 0 &&
                                parsedResult.toolCalls.length === 0
                              }
                              onClick={() => handleAppendAndContinue(true)}
                            >
                              <Send aria-hidden="true" /> Append & Send
                            </Button>
                          </div>
                        </div>
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="raw">
                    <pre
                      className="pg-code-box mono"
                      aria-label="Raw response payload"
                    >
                      {rawResponse || "No response payload captured yet."}
                    </pre>
                  </TabsContent>

                  <TabsContent value="sse">
                    <div className="pg-sse-list" aria-label="SSE frames">
                      {sseFrames.length === 0 ? (
                        <p className="muted">
                          No SSE frames recorded. Enable{" "}
                          <code>stream: true</code> and send a request.
                        </p>
                      ) : (
                        sseFrames.map((f) => (
                          <div key={f.index} className="pg-sse-frame mono">
                            <div className="pg-sse-meta">
                              <span>#{f.index}</span>
                              <span>+{f.elapsedMs}ms</span>
                              {f.event && (
                                <span className="violet">event: {f.event}</span>
                              )}
                            </div>
                            <pre>{f.data}</pre>
                          </div>
                        ))
                      )}
                    </div>
                  </TabsContent>

                  <TabsContent value="curl">
                    <pre className="pg-code-box mono" aria-label="cURL command">
                      {curlCommand}
                    </pre>
                  </TabsContent>
                </CardContent>
              </Tabs>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

function MethodBadge({ method }: { method: string }) {
  return (
    <Badge
      variant="outline"
      className={`font-mono text-[10px] ${method === "GET" ? "text-[var(--aqua)]" : "text-[var(--accent-bright)]"}`}
    >
      {method}
    </Badge>
  );
}

function ParameterBar({
  endpoint,
  body,
  maxTokensLimit,
  onUpdate,
  disabled,
}: {
  disabled: boolean;
  endpoint: EndpointSpec;
  body: Record<string, unknown>;
  maxTokensLimit: number;
  onUpdate: (
    updater: (b: Record<string, unknown>) => Record<string, unknown>,
  ) => void;
}) {
  const maxTokensKey =
    endpoint.kind === "openai-responses" ? "max_output_tokens" : "max_tokens";
  const maxTokensValue =
    typeof body[maxTokensKey] === "number"
      ? String(body[maxTokensKey] as number)
      : "";
  const temperature =
    typeof body.temperature === "number" ? body.temperature : 0;
  const stream = Boolean(body.stream);

  const thinkingEnabled =
    endpoint.kind === "openai-chat"
      ? Boolean(
          (body.chat_template_kwargs as Record<string, unknown> | undefined)
            ?.enable_thinking,
        )
      : endpoint.kind === "anthropic-messages"
        ? (body.thinking as Record<string, unknown> | undefined)?.type ===
          "enabled"
        : false;

  const minMaxTokens =
    endpoint.kind === "anthropic-messages" && thinkingEnabled ? 1025 : 1;

  return (
    <div className="pg-params-bar">
      <Field className="w-auto gap-2">
        <FieldLabel htmlFor="pg-max-tokens" className="eyebrow text-[11px]">
          {maxTokensKey.toUpperCase()}
        </FieldLabel>
        <Input
          id="pg-max-tokens"
          className="w-24"
          type="number"
          min={minMaxTokens}
          max={maxTokensLimit}
          placeholder="auto"
          value={maxTokensValue}
          onChange={(e) => {
            const raw = e.target.value.trim();
            if (raw === "" && endpoint.kind !== "anthropic-messages") {
              onUpdate((curr) => {
                const next = { ...curr };
                delete next[maxTokensKey];
                return next;
              });
              return;
            }
            const val = Math.min(
              maxTokensLimit,
              Math.max(minMaxTokens, Number(raw) || minMaxTokens),
            );
            onUpdate((curr) => ({ ...curr, [maxTokensKey]: val }));
          }}
        />
      </Field>

      <Field className="w-auto gap-2">
        <FieldLabel htmlFor="pg-temperature" className="eyebrow text-[11px]">
          TEMPERATURE
        </FieldLabel>
        <Input
          id="pg-temperature"
          className="w-24"
          type="number"
          min={0}
          max={2}
          step={0.1}
          value={temperature}
          onChange={(e) => {
            const val = Math.min(2, Math.max(0, Number(e.target.value) || 0));
            onUpdate((curr) => ({ ...curr, temperature: val }));
          }}
        />
      </Field>

      {endpoint.supportsStream && (
        <Field orientation="horizontal" className="w-auto gap-2 self-end pb-2">
          <Switch
            disabled={disabled}
            id="pg-stream"
            checked={stream}
            onCheckedChange={(checked) => {
              onUpdate((curr) => {
                const next: Record<string, unknown> = {
                  ...curr,
                  stream: checked,
                };
                if (endpoint.kind === "openai-chat") {
                  if (checked) {
                    next.stream_options = { include_usage: true };
                  } else {
                    delete next.stream_options;
                  }
                }
                return next;
              });
            }}
          />
          <FieldLabel htmlFor="pg-stream">Stream SSE</FieldLabel>
        </Field>
      )}

      {(endpoint.kind === "openai-chat" ||
        endpoint.kind === "anthropic-messages") && (
        <Field orientation="horizontal" className="w-auto gap-2 self-end pb-2">
          <Switch
            disabled={disabled}
            id="pg-thinking"
            checked={thinkingEnabled}
            onCheckedChange={(checked) => {
              onUpdate((curr) => {
                const next = { ...curr };
                if (endpoint.kind === "openai-chat") {
                  next.chat_template_kwargs = {
                    ...((curr.chat_template_kwargs as
                      Record<string, unknown> | undefined) ?? {}),
                    enable_thinking: checked,
                  };
                } else if (endpoint.kind === "anthropic-messages") {
                  if (checked) {
                    const currMax =
                      typeof curr.max_tokens === "number"
                        ? curr.max_tokens
                        : 256;
                    next.thinking = { type: "enabled", budget_tokens: 1024 };
                    next.max_tokens = Math.min(
                      maxTokensLimit,
                      Math.max(currMax, 4096),
                    );
                  } else {
                    next.thinking = { type: "disabled" };
                  }
                }
                return next;
              });
            }}
          />
          <FieldLabel htmlFor="pg-thinking">Thinking</FieldLabel>
        </Field>
      )}
    </div>
  );
}

function extractConditionImages(
  body: Record<string, unknown> | null,
): string[] {
  if (!body) return [];
  const out: string[] = [];
  for (const key of ["image", "images"] as const) {
    const val = body[key];
    if (typeof val === "string" && val.trim().length > 0) {
      out.push(val.trim());
    } else if (Array.isArray(val)) {
      for (const item of val) {
        if (typeof item === "string" && item.trim().length > 0) {
          out.push(item.trim());
        }
      }
    }
  }
  return out;
}

const RGBA_PROMPT_PREFIX = "This is an RGBA image with transparency. ";
const RGBA_PROMPT_SUFFIX =
  "The image has alpha channel and the background is transparent.";

function wrapRgbaTransparencyPrompt(prompt: string): string {
  const trimmed = prompt.trim();
  if (!trimmed) return `${RGBA_PROMPT_PREFIX}${RGBA_PROMPT_SUFFIX}`;
  const lower = trimmed.toLowerCase();
  if (lower.includes("rgba image") || lower.includes("alpha channel")) {
    return prompt;
  }
  const body = trimmed.replace(/\.+$/, "");
  return `${RGBA_PROMPT_PREFIX}${body}. ${RGBA_PROMPT_SUFFIX}`;
}

function unwrapRgbaTransparencyPrompt(prompt: string): string {
  let out = prompt.trim();
  if (out.toLowerCase().startsWith(RGBA_PROMPT_PREFIX.toLowerCase())) {
    out = out.slice(RGBA_PROMPT_PREFIX.length).trim();
  }
  if (out.toLowerCase().endsWith(RGBA_PROMPT_SUFFIX.toLowerCase())) {
    out = out
      .slice(0, out.length - RGBA_PROMPT_SUFFIX.length)
      .trim()
      .replace(/\.+$/, "");
  }
  return out;
}

const IMAGE_SIZE_OPTIONS = [
  "1024x1024",
  "512x512",
  "768x768",
  "1024x768",
  "768x1024",
  "auto",
] as const;

function ImageRequestEditor({
  endpoint,
  body,
  onUpdate,
  onAttachConditionImages,
  onRemoveConditionImage,
  onSubmit,
  disabled,
}: {
  endpoint: EndpointSpec;
  body: Record<string, unknown>;
  onUpdate: (
    updater: (b: Record<string, unknown>) => Record<string, unknown>,
  ) => void;
  onAttachConditionImages: (files: FileList | File[]) => Promise<void>;
  onRemoveConditionImage: (index: number) => void;
  onSubmit: () => void;
  disabled?: boolean;
}) {
  const isEditEndpoint = endpoint.id === "images-edits";
  const prompt = typeof body.prompt === "string" ? body.prompt : "";
  const negativePrompt =
    typeof body.negative_prompt === "string" ? body.negative_prompt : "";
  const size =
    typeof body.width === "number" && typeof body.height === "number"
      ? `${body.width}x${body.height}`
      : typeof body.size === "string"
        ? body.size
        : isEditEndpoint
          ? "auto"
          : "1024x1024";
  const steps =
    typeof body.steps === "number"
      ? body.steps
      : typeof body.num_inference_steps === "number"
        ? body.num_inference_steps
        : 8;
  const guidanceScale =
    typeof body.guidance_scale === "number"
      ? body.guidance_scale
      : typeof body.true_cfg_scale === "number"
        ? body.true_cfg_scale
        : typeof body.cfg_scale === "number"
          ? body.cfg_scale
          : 1.0;
  const seed = typeof body.seed === "number" ? body.seed : 42;
  const transparentBg = body.background === "transparent" || body.rgba === true;
  const conditionImages = extractConditionImages(body);

  return (
    <div className="pg-structured">
      <div className="pg-params-bar">
        <Field className="w-auto gap-2">
          <FieldLabel htmlFor="pg-img-size" className="eyebrow text-[11px]">
            SIZE
          </FieldLabel>
          <NativeSelect
            id="pg-img-size"
            disabled={disabled}
            aria-label="Image size"
            value={
              IMAGE_SIZE_OPTIONS.includes(
                size as (typeof IMAGE_SIZE_OPTIONS)[number],
              )
                ? size
                : "1024x1024"
            }
            onChange={(e) => {
              const nextSize = e.target.value;
              onUpdate((curr) => {
                const next: Record<string, unknown> = {
                  ...curr,
                  size: nextSize,
                };
                delete next.width;
                delete next.height;
                return next;
              });
            }}
          >
            {IMAGE_SIZE_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </NativeSelect>
        </Field>

        <Field className="w-auto gap-2">
          <FieldLabel htmlFor="pg-img-steps" className="eyebrow text-[11px]">
            STEPS
          </FieldLabel>
          <Input
            id="pg-img-steps"
            className="w-20"
            aria-label="Inference steps"
            type="number"
            min={1}
            max={100}
            value={steps}
            onChange={(e) => {
              const val = Math.min(
                100,
                Math.max(1, Number(e.target.value) || 8),
              );
              onUpdate((curr) => {
                const next: Record<string, unknown> = { ...curr, steps: val };
                delete next.num_inference_steps;
                return next;
              });
            }}
          />
        </Field>

        <Field className="w-auto gap-2">
          <FieldLabel htmlFor="pg-img-cfg" className="eyebrow text-[11px]">
            CFG SCALE
          </FieldLabel>
          <Input
            id="pg-img-cfg"
            className="w-20"
            aria-label="Guidance scale"
            type="number"
            min={0}
            max={30}
            step={0.5}
            value={guidanceScale}
            onChange={(e) => {
              const val = Math.min(
                30,
                Math.max(0, Number(e.target.value) || 0),
              );
              onUpdate((curr) => {
                const next: Record<string, unknown> = {
                  ...curr,
                  guidance_scale: val,
                };
                delete next.true_cfg_scale;
                delete next.cfg_scale;
                return next;
              });
            }}
          />
        </Field>

        <Field className="w-auto gap-2">
          <FieldLabel htmlFor="pg-img-seed" className="eyebrow text-[11px]">
            SEED
          </FieldLabel>
          <Input
            id="pg-img-seed"
            className="w-24"
            aria-label="Random seed"
            type="number"
            min={0}
            value={seed}
            onChange={(e) => {
              const val = Math.max(0, Number(e.target.value) || 0);
              onUpdate((curr) => ({ ...curr, seed: val }));
            }}
          />
        </Field>

        <Field orientation="horizontal" className="w-auto gap-2 self-end pb-2">
          <Switch
            disabled={disabled}
            id="pg-img-transparent"
            checked={transparentBg}
            onCheckedChange={(checked) => {
              onUpdate((curr) => {
                const currPrompt =
                  typeof curr.prompt === "string" ? curr.prompt : "";
                const nextPrompt =
                  !isEditEndpoint && currPrompt.length > 0
                    ? checked
                      ? wrapRgbaTransparencyPrompt(currPrompt)
                      : unwrapRgbaTransparencyPrompt(currPrompt)
                    : currPrompt;
                const next: Record<string, unknown> = {
                  ...curr,
                  prompt: nextPrompt,
                  background: checked ? "transparent" : "opaque",
                };
                delete next.rgba;
                return next;
              });
            }}
          />
          <FieldLabel htmlFor="pg-img-transparent">Transparent RGBA</FieldLabel>
        </Field>
      </div>

      {(isEditEndpoint || conditionImages.length > 0) && (
        <div
          className="pg-message-row"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            if (e.dataTransfer?.files?.length) {
              e.preventDefault();
              void onAttachConditionImages(e.dataTransfer.files);
            }
          }}
        >
          <div className="pg-message-meta">
            <span className="eyebrow text-[11px]">
              CONDITION IMAGES ({conditionImages.length}/
              {MAX_IMAGES_PER_REQUEST})
            </span>
            <div className="pg-turn-actions">
              {conditionImages.length < MAX_IMAGES_PER_REQUEST && (
                <Button
                  asChild
                  variant="outline"
                  size="sm"
                  className="focus-within:ring-2 focus-within:ring-ring"
                >
                  <label>
                    <Plus aria-hidden="true" /> Condition Image
                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/webp,image/gif"
                      multiple
                      aria-label="Attach condition image"
                      className="sr-only"
                      onChange={(e) => {
                        if (e.target.files && e.target.files.length > 0) {
                          void onAttachConditionImages(e.target.files);
                          e.target.value = "";
                        }
                      }}
                    />
                  </label>
                </Button>
              )}
            </div>
          </div>

          {conditionImages.length > 0 && (
            <div className="pg-turn-images">
              {conditionImages.map((imgUrl, imgIdx) => (
                <div key={imgIdx} className="pg-turn-thumb">
                  <img
                    src={imgUrl}
                    alt={`Condition image ${imgIdx + 1}`}
                    className="pg-turn-thumb-img"
                  />
                  <span className="pg-turn-thumb-meta mono">
                    IMG #{imgIdx + 1}
                  </span>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon-xs"
                    className="pg-thumb-remove"
                    aria-label={`Remove condition image ${imgIdx + 1}`}
                    onClick={() => onRemoveConditionImage(imgIdx)}
                  >
                    <X aria-hidden="true" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="pg-messages-list">
        <div
          className="pg-message-row"
          onPaste={(e) => {
            if (isEditEndpoint && e.clipboardData?.files?.length) {
              const imgFiles = Array.from(e.clipboardData.files).filter((f) =>
                f.type.startsWith("image/"),
              );
              if (imgFiles.length > 0) {
                e.preventDefault();
                void onAttachConditionImages(imgFiles);
              }
            }
          }}
        >
          <div className="pg-message-meta">
            <span className="eyebrow text-[11px]">PROMPT</span>
          </div>
          <Textarea
            aria-label="Image prompt"
            rows={4}
            value={prompt}
            placeholder="Describe the image to generate or edit…"
            onChange={(e) => {
              const val = e.target.value;
              onUpdate((curr) => ({ ...curr, prompt: val }));
            }}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                onSubmit();
              }
            }}
          />
        </div>

        <div className="pg-message-row">
          <div className="pg-message-meta">
            <span className="eyebrow text-[11px]">
              NEGATIVE PROMPT (OPTIONAL, CFG &gt; 1)
            </span>
          </div>
          <Textarea
            aria-label="Negative prompt"
            rows={2}
            value={negativePrompt}
            placeholder="Optional negative prompt when guidance_scale > 1.0…"
            onChange={(e) => {
              const val = e.target.value;
              onUpdate((curr) => {
                const next = { ...curr };
                if (val.trim().length > 0) {
                  next.negative_prompt = val;
                } else {
                  delete next.negative_prompt;
                }
                return next;
              });
            }}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                onSubmit();
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}
