document.addEventListener('DOMContentLoaded', function () {
    const btn = document.getElementById('toggleTextBtn');
    const container = document.getElementById('jupyterContainer');

    if (btn && container) {
        btn.addEventListener('click', function () {
            container.classList.toggle('full-text-mode');

            if (container.classList.contains('full-text-mode')) {
                btn.innerHTML = 'Recolher';
                btn.classList.replace('btn-outline-secondary', 'btn-secondary');
            } else {
                btn.innerHTML = 'Expandir Texto';
                btn.classList.replace('btn-secondary', 'btn-outline-secondary');
            }
        });
    } else {
        console.error('Botão #toggleTextBtn ou container #jupyterContainer não foram encontrados no DOM.');
    }
});