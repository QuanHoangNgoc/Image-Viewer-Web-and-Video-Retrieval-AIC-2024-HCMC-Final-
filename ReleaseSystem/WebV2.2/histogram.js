
function showPie(values) {
    // Count occurrences of each value
    const counts = {};
    values.forEach((value, index) => {
        let score = 0.0; // Initialize score as a float
        if (index < 1) score += 0.2; 
        if (index < 5) score += 0.2; 
        if (index < 20) score += 0.2; 
        if (index < 50) score += 0.2; 
        if(index < 100) score += 0.2; 
        counts[value] = (counts[value] || 0) + score; 
    });
    
    // Prepare data for the pie chart
    let labels = Object.keys(counts);
    let data = Object.values(counts);
    
    // Create an array of [label, value] pairs
    const combined = labels.map((label, index) => [label, data[index]]);
    // Sort the array based on the value (second element of the pair) in descending order
    combined.sort((a, b) => b[1] - a[1]); // Sort by the second element (value)
    // Separate the sorted labels and data back into individual arrays
    labels = combined.map(pair => pair[0]);
    data = combined.map(pair => pair[1]);
    
    // Show the canvas
    const canvas = document.getElementById('myPieChart');
    const ctx = canvas.getContext('2d');
    canvas.style.display = 'block';  // Make canvas visible
    
    // Create the pie chart
    new Chart(ctx, {
        type: 'pie',  // Type of chart
        data: {
            labels: labels,  // Unique values as labels
            datasets: [{
                label: 'Frequency',
                data: data,  // Counts of each value
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    enabled: true
                }
            }
        }
    });
}

function showHistogram(values) {
    // Count occurrences of each value
    const counts = {};
    values.forEach(value => {
        counts[value] = (counts[value] || 0) + 1;
    });
    
    // Prepare data for the histogram
    const labels = Object.keys(counts);
    const data = Object.values(counts);
    
    // Show the canvas
    const canvas = document.getElementById('myHistogram');
    const ctx = canvas.getContext('2d');
    canvas.style.display = 'block';  // Make canvas visible
    
    // Create the histogram
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,  // Unique values as labels
            datasets: [{
                label: 'Frequency',
                data: data,  // Counts of each value
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}