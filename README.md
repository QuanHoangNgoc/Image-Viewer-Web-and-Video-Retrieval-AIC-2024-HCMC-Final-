# 📷 ***CSV Image Viewer & Search Engine***

![image](https://user-images.githubusercontent.com/xx/xx/image.png)

## 🔍 What is it?
This project implements a CSV-Based Image Viewer and Search Engine that allows users to search images based on text queries or specific components. Users can explore and filter images from a dataset using a customized, simple web interface.

## 🤔 Why did we build it?
This project was built to make large-scale image datasets easier to browse and search. It enables users to search using natural language descriptions, match visual-textual components, and filter results by multiple criteria.

## 👥 Who can use it?
- **Researchers** in AI/ML looking for a user-friendly interface to explore datasets.
- **Developers** are working with image datasets that need to test and visualize image search functionalities.
- **Users** with an interest in simple and accessible image exploration tools.

**Demo**: [Click here for a live demo](#)

### Key Results
- **Search by text queries:** Input descriptions to find the most relevant images.
- **Component-based search:** Compare visual features or text features across images.
  
## 🛠️ How did we build it?
1. **Image dataset handling**: Processed datasets to retrieve, visualize, and explore-exploit for images.
2. **Embeddings & FAISS**: Implemented FAISS to create an image vector database.
3. **Translation and Encoding**: Used `googletrans` for multi-language support and `BEiT3` for text/image embeddings.
4. **Web interface**: Built with **JavaScript** to upload CSV files and dynamically show, explore, and manage images.

### Detailed Steps:
- **Image processing**: We used Python libraries such as `PIL` and `Faiss` to handle image files and embeddings.
- **Embedding generation**: Leveraged `BEiT3` to encode images and text into a searchable vector format.
- **Vector Search**: Built a **FAISS**-based vector database to allow fast search on large datasets.
- **UI Interaction**: JavaScript-based web interface to explore, display images, and export results as CSV.

## 📚 What did we learn?
- **Efficient Image Retrieval**: How to scale image retrieval using FAISS.
- **Text & Image Search**: Combining text descriptions with image embeddings leads to accurate search results.
- **UI Optimization**: Creating a seamless user experience for dataset exploration.

## 🏆 Achievements
- **Optimized Search**: Fast search for large-scale image datasets with multiple query options.
- **Multi-language Support**: Translate non-English queries for international datasets.
- **User-friendly Web Interface**: Easy-to-use platform for researchers and developers.

## 🤝 Contributing
Feel free to fork this repository and submit pull requests. Contributions are always welcome!

## ⭐ Support & Donations
If you like this project, please **give it a star** on GitHub! ⭐
**Author**: Quan-Hoang-Ngoc  
