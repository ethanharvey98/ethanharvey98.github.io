function includeHTML() {
    document.querySelectorAll("[html]").forEach(element => {
        fetch(element.getAttribute("html"))
        .then(response => response.text())
        .then(html => {
            element.innerHTML = html;
            element.removeAttribute("html");
        });
    });
}

includeHTML();
