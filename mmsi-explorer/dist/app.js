const state = { records: [], filtered: [], category: 'all', difficulty: 'all', query: '', sort: 'id', page: 1 };
const pageSize = 12;
const els = {
  totalCount: document.querySelector('#totalCount'), resultCount: document.querySelector('#resultCount'),
  categoryFilters: document.querySelector('#categoryFilters'), difficultyFilters: document.querySelector('#difficultyFilters'),
  search: document.querySelector('#search'), showAnswers: document.querySelector('#showAnswers'), sort: document.querySelector('#sort'),
  status: document.querySelector('#status'), grid: document.querySelector('#grid'), template: document.querySelector('#cardTemplate'),
  prev: document.querySelector('#prevPage'), next: document.querySelector('#nextPage'), pageLabel: document.querySelector('#pageLabel'),
  dialog: document.querySelector('#detailDialog'), detail: document.querySelector('#detailContent'), close: document.querySelector('#closeDialog')
};

const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
const optionText = (question, letter) => {
  const match = question.match(new RegExp(`(?:Options:\\s*)?${letter}:\\s*(.*?)(?=,?\\s+[A-D]:|$)`, 'i'));
  return match ? match[1].trim() : '';
};

async function loadData() {
  try {
    const response = await fetch('records.json');
    if (!response.ok) throw new Error('Dataset file unavailable');
    state.records = await response.json();
    els.totalCount.textContent = state.records.length.toLocaleString();
    buildCategoryFilters();
    applyFilters();
  } catch (error) {
    els.status.textContent = 'Could not load the local dataset. Start the included viewer server and refresh.';
  }
}

function buildCategoryFilters() {
  const categories = [...new Set(state.records.map(r => r.category))].sort();
  els.categoryFilters.innerHTML = ['all', ...categories].map(category =>
    `<button class="chip ${category === 'all' ? 'active' : ''}" data-category="${escapeHtml(category)}">${category === 'all' ? 'All categories' : escapeHtml(category)}</button>`
  ).join('');
}

function applyFilters() {
  const q = state.query.toLowerCase();
  state.filtered = state.records.filter(r =>
    (state.category === 'all' || r.category === state.category) &&
    (state.difficulty === 'all' || r.difficulty === state.difficulty) &&
    (!q || `${r.question} ${r.thought}`.toLowerCase().includes(q))
  );
  const rank = { easy: 0, medium: 1, hard: 2 };
  state.filtered.sort((a, b) => {
    if (state.sort === 'hardest') return rank[b.difficulty] - rank[a.difficulty] || a.id - b.id;
    if (state.sort === 'easiest') return rank[a.difficulty] - rank[b.difficulty] || a.id - b.id;
    if (state.sort === 'slowest') return (b.humanTime ?? -99) - (a.humanTime ?? -99);
    return a.id - b.id;
  });
  state.page = Math.min(state.page, Math.max(1, Math.ceil(state.filtered.length / pageSize)));
  render();
}

function render() {
  els.resultCount.textContent = state.filtered.length.toLocaleString();
  els.status.textContent = state.filtered.length ? '' : 'No questions match these filters.';
  els.grid.innerHTML = '';
  const start = (state.page - 1) * pageSize;
  state.filtered.slice(start, start + pageSize).forEach(record => {
    const node = els.template.content.cloneNode(true);
    const card = node.querySelector('.question-card');
    node.querySelector('.question-id').textContent = `#${String(record.id).padStart(4, '0')}`;
    const pill = node.querySelector('.difficulty-pill');
    pill.textContent = record.difficulty;
    pill.classList.add(record.difficulty);
    node.querySelector('.category').textContent = record.category;
    node.querySelector('.question').textContent = record.question;
    node.querySelector('.answer-preview').textContent = `Answer ${record.answer}: ${optionText(record.question, record.answer)}`;
    node.querySelector('.thumb-strip').innerHTML = record.images.slice(0, 4).map((src, i) => `<img src="${src}" alt="Question ${record.id}, image ${i + 1}" loading="lazy">`).join('');
    card.querySelector('.open-detail').addEventListener('click', () => openDetail(record));
    els.grid.appendChild(node);
  });
  const pages = Math.max(1, Math.ceil(state.filtered.length / pageSize));
  els.pageLabel.textContent = `Page ${state.page} of ${pages}`;
  els.prev.disabled = state.page <= 1;
  els.next.disabled = state.page >= pages;
}

function openDetail(record) {
  els.detail.innerHTML = `
    <div class="detail-meta"><strong>#${String(record.id).padStart(4, '0')}</strong><span>·</span><span>${escapeHtml(record.category)}</span><span>·</span><span>${escapeHtml(record.difficulty)}</span></div>
    <h2 class="detail-question">${escapeHtml(record.question)}</h2>
    <div class="detail-images">${record.images.map((src, i) => `<img src="${src}" alt="Question ${record.id}, image ${i + 1}">`).join('')}</div>
    <section class="detail-section"><h3>Correct answer</h3><div class="correct-answer">${escapeHtml(record.answer)}: ${escapeHtml(optionText(record.question, record.answer))}</div></section>
    <section class="detail-section"><h3>Reference reasoning</h3><div class="rationale">${escapeHtml(record.thought)}</div></section>`;
  els.dialog.showModal();
}

function registerWebMcp() {
  const context = document.modelContext;
  if (!context?.registerTool) return;
  try {
    void Promise.resolve(context.registerTool({
      name: 'filter_questions',
      title: 'Filter MMSI-Bench questions',
      description: 'Filter the visible dataset by exact category, difficulty, or text search and return the number of matching questions.',
      inputSchema: {
        type: 'object',
        properties: {
          category: { type: 'string', description: 'Exact category label, or all.' },
          difficulty: { type: 'string', enum: ['all', 'easy', 'medium', 'hard'] },
          query: { type: 'string' }
        },
        additionalProperties: false
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute(input = {}) {
        const categories = new Set(['all', ...state.records.map(record => record.category)]);
        if (input.category !== undefined && !categories.has(input.category)) throw new Error('Unknown category');
        if (input.category !== undefined) state.category = input.category;
        if (input.difficulty !== undefined) state.difficulty = input.difficulty;
        if (input.query !== undefined) state.query = String(input.query).trim();
        state.page = 1;
        els.search.value = state.query;
        els.categoryFilters.querySelectorAll('.chip').forEach(el => el.classList.toggle('active', el.dataset.category === state.category));
        els.difficultyFilters.querySelectorAll('.chip').forEach(el => el.classList.toggle('active', el.dataset.difficulty === state.difficulty));
        applyFilters();
        return { matchingQuestions: state.filtered.length, category: state.category, difficulty: state.difficulty, query: state.query };
      }
    }));
  } catch (error) {
    console.debug('WebMCP unavailable', error);
  }
}

els.categoryFilters.addEventListener('click', event => {
  const button = event.target.closest('[data-category]'); if (!button) return;
  state.category = button.dataset.category; state.page = 1;
  els.categoryFilters.querySelectorAll('.chip').forEach(el => el.classList.toggle('active', el === button)); applyFilters();
});
els.difficultyFilters.addEventListener('click', event => {
  const button = event.target.closest('[data-difficulty]'); if (!button) return;
  state.difficulty = button.dataset.difficulty; state.page = 1;
  els.difficultyFilters.querySelectorAll('.chip').forEach(el => el.classList.toggle('active', el === button)); applyFilters();
});
els.search.addEventListener('input', () => { state.query = els.search.value.trim(); state.page = 1; applyFilters(); });
els.sort.addEventListener('change', () => { state.sort = els.sort.value; state.page = 1; applyFilters(); });
els.showAnswers.addEventListener('change', () => document.body.classList.toggle('answers-visible', els.showAnswers.checked));
els.prev.addEventListener('click', () => { state.page--; render(); scrollTo({ top: 0, behavior: 'smooth' }); });
els.next.addEventListener('click', () => { state.page++; render(); scrollTo({ top: 0, behavior: 'smooth' }); });
els.close.addEventListener('click', () => els.dialog.close());
els.dialog.addEventListener('click', event => { if (event.target === els.dialog) els.dialog.close(); });
loadData().then(registerWebMcp);
