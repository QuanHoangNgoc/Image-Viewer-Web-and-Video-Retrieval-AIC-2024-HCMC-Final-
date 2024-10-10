function downloadCSV(csvText, filename) {
  // Create a blob from the CSV text
  const blob = new Blob([csvText], { type: 'text/csv' });
  
  // Create a link element
  const link = document.createElement('a');
  
  // Create a URL for the blob and set it as the href attribute
  link.href = window.URL.createObjectURL(blob);
  
  // Set the download attribute to the filename
  link.download = filename;
  
  // Programmatically click the link to trigger the download
  link.click();
}