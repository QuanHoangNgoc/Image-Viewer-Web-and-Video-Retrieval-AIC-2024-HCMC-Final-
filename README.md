# 📷 ***CSV Image Viewer & Search Engine*** (AIC 2024 HCMC Final)

## 🔍 What is it?
This project implements a CSV-Based Image Viewer and Search Engine that allows users to search images based on text queries or specific components. Users can explore and manage images from a dataset using a customized, simple web interface.

## 🤔 Why did we build it?
This project was built to make large-scale image datasets easier to browse and search. It enables users to search using natural language descriptions, match visual-textual components, and filter results by multiple criteria.

## 👥 Who can use it?
- **Researchers** in AI/ML looking for a user-friendly interface to explore datasets.
- **Developers** are working with image datasets that need to test and visualize image search functionalities.
- **Users** with an interest in simple and accessible image exploration tools.

**Demo**: [Click here for a live demo](#)
### Ranking

<img src="https://github.com/user-attachments/assets/391d6ae8-ef71-42dc-a034-1abc5f7a35b3" alt="visual" width="300"/>
<img src="https://github.com/user-attachments/assets/bdeab9de-4f64-4eff-a565-eec56999dc5c" alt="visual" width="300"/>
<img src="https://github.com/user-attachments/assets/9a051e57-ce10-454e-a665-4985e33d72c1" alt="visual" width="300"/>

- ***Note***: When I joined, our ranking rose to **top 10**. This is because I developed advanced image search options based on my knowledge and learning (two-phase search, element-centered search, conscious search) that increased precision and recall for faster image discovery and exploitation. Additionally, I also designed the UI, implemented the web using **JavaScript**, and tested it to ensure a smooth, user-friendly experience for the team.

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
