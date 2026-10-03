const storageKey = "fvpg-overlap-labels-v1";
let sample = null;
let current = 0;
let labels = JSON.parse(localStorage.getItem(storageKey) || "{}");

const $ = (id) => document.getElementById(id);

function save() { localStorage.setItem(storageKey, JSON.stringify(labels)); }
function selected() { return labels[sample.items[current].key] || {}; }

function render() {
  const item = sample.items[current];
  const value = selected();
  $("imageA").src = item.images[0];
  $("imageB").src = item.images[1];
  $("position").textContent = `${current + 1} / ${sample.items.length}`;
  const done = Object.values(labels).filter(v => v.relationship).length;
  $("progressText").textContent = `· ${done} labeled`;
  $("progressBar").style.width = `${100 * done / sample.items.length}%`;
  document.querySelectorAll("#relationshipLabels button").forEach(button => {
    button.classList.toggle("selected", button.dataset.value === value.relationship);
  });
  $("fractionField").disabled = value.relationship !== "overlap";
  document.querySelectorAll("#fractionLabels button").forEach(button => {
    button.classList.toggle("selected", button.dataset.value === value.fraction);
  });
  $("notes").value = value.notes || "";
  $("previous").disabled = current === 0;
}

function update(patch) {
  const key = sample.items[current].key;
  labels[key] = { ...labels[key], ...patch, updated_at: new Date().toISOString() };
  if (patch.relationship && patch.relationship !== "overlap") labels[key].fraction = null;
  save(); render();
}

function nextUnlabeled() {
  for (let offset = 1; offset <= sample.items.length; offset++) {
    const index = (current + offset) % sample.items.length;
    if (!labels[sample.items[index].key]?.relationship) { current = index; render(); return; }
  }
  current = Math.min(current + 1, sample.items.length - 1); render();
}

document.querySelectorAll("#relationshipLabels button").forEach(button => button.addEventListener("click", () => update({ relationship: button.dataset.value })));
document.querySelectorAll("#fractionLabels button").forEach(button => button.addEventListener("click", () => update({ fraction: button.dataset.value })));
$("notes").addEventListener("change", event => update({ notes: event.target.value.trim() }));
$("previous").addEventListener("click", () => { current = Math.max(0, current - 1); render(); });
$("next").addEventListener("click", nextUnlabeled);
$("export").addEventListener("click", () => {
  const annotations = sample.items.map(item => ({ ...item, annotation: labels[item.key] || null }));
  const blob = new Blob([JSON.stringify({ schema_version: 1, exported_at: new Date().toISOString(), annotations }, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "mmsi_overlap_annotations.json"; link.click();
  URL.revokeObjectURL(link.href);
});
document.addEventListener("keydown", event => {
  if (event.target.tagName === "TEXTAREA") return;
  const values = ["overlap", "same_scene_no_overlap", "different_scene", "uncertain"];
  if (/^[1-4]$/.test(event.key)) update({ relationship: values[Number(event.key) - 1] });
  if (event.key === "ArrowRight") nextUnlabeled();
  if (event.key === "ArrowLeft") { current = Math.max(0, current - 1); render(); }
});

fetch("overlap_annotation_sample.json")
  .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then(data => { sample = data; $("status").hidden = true; $("workspace").hidden = false; render(); })
  .catch(error => { $("status").textContent = `Could not load annotation sample: ${error.message}`; });
