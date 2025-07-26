
import 'dart:io';

import 'package:flutter/material.dart';

import '../../data/models/album.dart';
import 'image_viewer_screen.dart';

class AlbumDetailScreen extends StatelessWidget {
  final Album album;

  const AlbumDetailScreen({super.key, required this.album});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          album.commonContextTheme.isNotEmpty 
              ? album.commonContextTheme 
              : album.name,
        ),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${album.imageCount} images',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                if (album.commonObjectLabel.isNotEmpty)
                  Text('Objects: ${album.commonObjectLabel}'),
                if (album.facesCategories.isNotEmpty)
                  Text('Faces: ${album.facesCategories}'),
                if (album.dominantFolder.isNotEmpty)
                  Text('Main folder: ${album.dominantFolder}'),
              ],
            ),
          ),
          Expanded(
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 4,
                crossAxisSpacing: 4,
                mainAxisSpacing: 4,
              ),
              itemCount: album.imagePaths.length,
              itemBuilder: (context, index) {
                return GestureDetector(
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ImageViewerScreen(
                          images: album.imagePaths,
                          initialIndex: index,
                        ),
                      ),
                    );
                  },
                  child: Image.file(
                    File(album.imagePaths[index]),
                    fit: BoxFit.cover,
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

