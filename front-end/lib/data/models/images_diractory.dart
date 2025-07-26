import 'dart:io';
import 'dart:math';

const String imagesDirectoryPath = "C:\\Users\\DELL\\Desktop\\unsorted_shit\\daily_life_photography";

List<String> getHomeImagesBackground() {
  final directory = Directory(imagesDirectoryPath);
  List<String> imagePaths = [];
  
  if (directory.existsSync()) {
    List<FileSystemEntity> files = directory.listSync();
    
    List<FileSystemEntity> imageFiles = files.where((file) {
      String path = file.path.toLowerCase();
      return path.endsWith('.jpg') || 
             path.endsWith('.jpeg') || 
             path.endsWith('.png');
    }).toList();
    
    imageFiles.shuffle();
    
    int count = min(4, imageFiles.length);
    for (int i = 0; i < count; i++) {
      imagePaths.add(imageFiles[i].path);
    }
  }
  
  return imagePaths;
}