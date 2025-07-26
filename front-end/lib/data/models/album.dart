class Album {
  final String name;
  final List<String> imagePaths;
  final String facesCategories;
  final String commonObjectLabel;
  final String dominantFolder;
  final String commonBackgroundClass;
  final String commonContextTheme;

  Album({
    required this.name,
    required this.imagePaths,
    required this.facesCategories,
    required this.commonObjectLabel,
    required this.dominantFolder,
    required this.commonBackgroundClass,
    required this.commonContextTheme,
  });

  factory Album.fromJson(String name, Map<String, dynamic> json) {
    return Album(
      name: name,
      imagePaths: List<String>.from(json['image_paths'] ?? []),
      facesCategories: json['faces_categories'] ?? '',
      commonObjectLabel: json['common_object_label'] ?? '',
      dominantFolder: json['dominant_folder'] ?? '',
      commonBackgroundClass: json['common_background_class'] ?? '',
      commonContextTheme: json['common_context_theme'] ?? '',
    );
  }

  int get imageCount => imagePaths.length;
}