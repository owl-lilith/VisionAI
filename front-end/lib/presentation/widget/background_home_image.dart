import 'dart:io';
import 'package:flutter/material.dart';

class BackgroundHomeImage extends StatelessWidget {
  const BackgroundHomeImage({
    super.key,
    required this.bg_images,
    required this.bg_images_index,
    required this.alignment,
  });

  final List<String> bg_images;
  final int bg_images_index;
  final Alignment alignment;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.all(10),
      alignment: alignment,

      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.file(File(bg_images[bg_images_index]), height: 200),
      ),
    );
  }
}
