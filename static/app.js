const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const button = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
const metrics = document.querySelector("#metrics");
let processing = false;

const setMessage = (text, isError = false) => {
  message.textContent = text;
  message.classList.toggle("error", isError);
};

const formatFileSize = (bytes) => `${(bytes / (1024 * 1024)).toFixed(bytes >= 1024 * 1024 ? 1 : 2)} MB`;

fileInput.addEventListener("change", () => {
  if (processing) return;
  const file = fileInput.files[0];
  if (file) {
    setMessage(`Image selected: ${file.name} (${formatFileSize(file.size)}). Ready to generate the mask.`);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (processing) return;

  const file = fileInput.files[0];
  if (!file) {
    setMessage("Pick an H&E image first.", true);
    return;
  }

  const body = new FormData();
  body.append("file", file);
  processing = true;
  button.disabled = true;
  fileInput.disabled = true;
  button.textContent = "Processing image…";
  button.setAttribute("aria-busy", "true");
  setMessage("Image uploaded. Generating the segmentation mask—please wait, uploads are locked until processing finishes.");

  try {
    const response = await fetch("/predict", { method: "POST", body });
    const raw = await response.text();
    let data = {};
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      data = { detail: raw };
    }
    if (!response.ok) throw new Error(data.detail || `Prediction failed with HTTP ${response.status}. Check Render logs for the backend traceback.`);

    document.querySelector("#input-img").src = data.input;
    document.querySelector("#mask-img").src = data.mask;
    document.querySelector("#prob-img").src = data.probability;
    document.querySelector("#overlay-img").src = data.overlay;
    document.querySelector("#foreground").textContent = `${data.foreground_percent}%`;
    results.classList.remove("hidden");
    metrics.classList.remove("hidden");
    setMessage("Segmentation complete.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    processing = false;
    button.disabled = false;
    fileInput.disabled = false;
    button.textContent = "Generate segmentation";
    button.removeAttribute("aria-busy");
  }
});
