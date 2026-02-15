// Load books on page load
if (document.getElementById('bookList')) {
    fetchBooks();
}

async function fetchBooks() {
    const response = await fetch('/api/books');
    const books = await response.json();
    const container = document.getElementById('bookList');
    container.innerHTML = '';
    
    books.forEach(book => {
        const div = document.createElement('div');
        div.className = 'book-card';
        div.innerHTML = `
            <h4>${book.title}</h4>
            <p><strong>Genre:</strong> ${book.genre}</p>
            <p><em>Tags: ${book.tags}</em></p>
            <button onclick="borrowBook('${book.title}')">Borrow</button>
        `;
        container.appendChild(div);
    });
}

// Simulate Borrowing & Trigger Recommendations
function borrowBook(title) {
    alert(`You borrowed: ${title}`);
    // Get recommendations based on this interaction (Hybrid Trigger)
    getRecommendations(title); 
}

async function getRecommendations(lastBook = null) {
    // Mocking User ID 1 for demonstration
    const payload = { user_id: 1, last_book: lastBook };
    
    const response = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    
    const recs = await response.json();
    const list = document.getElementById('recList');
    list.innerHTML = '';
    
    if (recs.length === 0) {
        list.innerHTML = '<li>Borrow a book to get started!</li>';
    } else {
        recs.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            list.appendChild(li);
        });
    }
}

// Admin Charts [cite: 76]
async function loadAdminCharts() {
    const response = await fetch('/api/stats');
    const data = await response.json();

    // Department Usage Chart
    const ctxDept = document.getElementById('deptChart').getContext('2d');
    new Chart(ctxDept, {
        type: 'pie',
        data: {
            labels: Object.keys(data.dept_usage),
            datasets: [{
                data: Object.values(data.dept_usage),
                backgroundColor: ['#e74c3c', '#3498db', '#f1c40f', '#2ecc71']
            }]
        }
    });

    // Hourly Traffic Chart
    const ctxHour = document.getElementById('hourChart').getContext('2d');
    new Chart(ctxHour, {
        type: 'line',
        data: {
            labels: ['8am', '10am', '12pm', '2pm', '4pm'],
            datasets: [{
                label: 'Library Visitors',
                data: data.peak_hours,
                borderColor: '#2c3e50',
                fill: false
            }]
        }
    });
}