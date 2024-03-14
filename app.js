const updateLastUpdated = async () => {
    try {
        const response = await fetch('https://api.github.com/repos/ethanharvey98/ethanharvey98.github.io/events');
        const data = await response.json();
        const pushEvent = data.find(event => event.type === 'PushEvent');
        if (pushEvent) {
            const lastUpdated = new Date(pushEvent.created_at);
            const options = { year: 'numeric', month: 'long', day: 'numeric' };
            const formattedDate = lastUpdated.toLocaleDateString(undefined, options);
            document.getElementById('last-updated').textContent = `Last Updated: ${formattedDate}`;
        }
    } catch (error) {
        console.error(error);
    }
};

updateLastUpdated();

fetch('news.md')
.then(response => {
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.text();
})
.then(mdContent => {
    const htmlContent = marked(mdContent);
    document.querySelector('#news').innerHTML = htmlContent;
})
.catch(error => {
    console.error('Error fetching Markdown file:', error);
});