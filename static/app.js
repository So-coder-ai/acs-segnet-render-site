const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const button = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const results = document.querySelector("#results");
const metrics = document.querySelector("#metrics");

const setMessage = (text, isError = false) => {
  message.textContent = text;
  message.classList.toggle("error", isError);
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setMessage("Pick an H&E image first.", true);
    return;
  }

  const body = new FormData();
  body.append("file", file);
  button.disabled = true;
  setMessage("Running ACS-SegNet inference… this can take a moment on CPU.");

  try {
    const response = await fetch("/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Prediction failed.");

    document.querySelector("#input-img").src = data.input;
    document.querySelector("#mask-img").src = data.mask;
    document.querySelector("#prob-img").src = data.probability;
    document.querySelector("#overlay-img").src = data.overlay;
    document.querySelector("#foreground").textContent = `${data.foreground_percent}%`;
    results.classList.remove("hidden");
    metrics.classList.remove("hidden");
    setMessage("Segmentation complete. Little red mask goblin deployed successfully.");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    button.disabled = false;
  }
});
