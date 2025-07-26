import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';

import '../../data/models/album.dart';
import 'album_view_details_screen.dart';

class AlbumScreen extends StatefulWidget {
  static String routeName = '/album-screen';
  const AlbumScreen({super.key});

  @override
  State<AlbumScreen> createState() => _AlbumScreenState();
}

class _AlbumScreenState extends State<AlbumScreen> {
  Map<String, Album> albums = {};
  bool isLoading = true;
  final Map<String, int> _currentImageIndex = {};
  final Map<String, Timer?> _slideTimers = {};
  final Map<String, List<String>> _shuffledImages = {};

  @override
  void initState() {
    super.initState();
    _loadAlbums();
  }

  @override
  void dispose() {
    // Cancel all timers when widget is disposed
    for (var timer in _slideTimers.values) {
      timer?.cancel();
    }
    super.dispose();
  }

  Future<void> _loadAlbums() async {
    try {
      final file = File(
        'D:/image_search_engine_ai-end/sources/metadata/albums_data.json',
      );
      final jsonString = await file.readAsString();
      final jsonData = json.decode(jsonString) as Map<String, dynamic>;

      setState(() {
        albums = _filterAlbums(
          jsonData.map(
            (name, data) => MapEntry(name, Album.fromJson(name, data)),
          ),
        );
        isLoading = false;

        // Initialize shuffled images and current index for each album
        for (var album in albums.values) {
          _shuffledImages[album.name] = List.from(album.imagePaths)..shuffle();
          _currentImageIndex[album.name] = 0;
        }
      });
    } catch (e) {
      setState(() => isLoading = false);
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Error loading albums: $e')));
    }
  }

  Map<String, Album> _filterAlbums(Map<String, Album> allAlbums) {
    const minImages = 5;
    return Map.fromEntries(
      allAlbums.entries.where(
        (entry) => entry.key != 'Noise' && entry.value.imageCount >= minImages,
      ),
    );
  }

  void _startSlideshow(String albumName) {
    // Cancel any existing timer for this album
    _slideTimers[albumName]?.cancel();

    // Start new timer
    _slideTimers[albumName] = Timer.periodic(const Duration(seconds: 2), (
      timer,
    ) {
      if (mounted) {
        setState(() {
          final currentIndex = _currentImageIndex[albumName] ?? 0;
          final images = _shuffledImages[albumName] ?? [];
          _currentImageIndex[albumName] = (currentIndex + 1) % images.length;
        });
      }
    });
  }

  void _stopSlideshow(String albumName) {
    _slideTimers[albumName]?.cancel();
    _slideTimers[albumName] = null;
  }

  @override
  Widget build(BuildContext context) {
    if (isLoading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Photo Albums'), centerTitle: true),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.builder(
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 3,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
            childAspectRatio: 0.8,
          ),
          itemCount: albums.length,
          itemBuilder: (context, index) {
            final album = albums.values.elementAt(index);
            return _buildAlbumCard(album, context);
          },
        ),
      ),
    );
  }

  Widget _buildAlbumCard(Album album, BuildContext context) {
    return MouseRegion(
      onEnter: (_) {
        _shuffledImages[album.name] = List.from(album.imagePaths)..shuffle();
        _currentImageIndex[album.name] = 0;
        _startSlideshow(album.name);
      },
      onExit: (_) {
        _stopSlideshow(album.name);
        setState(() {
          _currentImageIndex[album.name] = 0;
        });
      },
      child: GestureDetector(
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => AlbumDetailScreen(album: album),
            ),
          );
        },
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeInOut,
          transform: Matrix4.identity(),
          child: Card(
            clipBehavior: Clip.antiAlias,
            elevation: 8,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Stack(
              fit: StackFit.expand,
              children: [
                // Slideshow image
                _buildSlideshowImage(album),

                // Album name label at top
                Positioned(
                  top: 0,
                  left: 0,
                  right: 0,
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.black.withOpacity(0.7),
                          Colors.transparent,
                        ],
                      ),
                    ),
                    child: Text(
                      album.commonContextTheme.isNotEmpty
                          ? album.commonContextTheme
                          : album.name,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ),

                // Info footer
                Positioned(
                  bottom: 0,
                  left: 0,
                  right: 0,
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.bottomCenter,
                        end: Alignment.topCenter,
                        colors: [
                          Colors.black.withOpacity(0.8),
                          Colors.transparent,
                        ],
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '${album.imageCount} images',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSlideshowImage(Album album) {
    final images = _shuffledImages[album.name] ?? [];
    final currentIndex = _currentImageIndex[album.name] ?? 0;
    final imagePath = images.isNotEmpty ? images[currentIndex] : '';

    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 500),
      transitionBuilder: (Widget child, Animation<double> animation) {
        return FadeTransition(opacity: animation, child: child);
      },
      child:
          imagePath.isNotEmpty
              ? Image.file(
                File(imagePath),
                key: ValueKey<String>(imagePath),
                fit: BoxFit.cover,
                errorBuilder:
                    (context, error, stackTrace) => Container(
                      color: Colors.grey[300],
                      child: const Icon(Icons.broken_image),
                    ),
              )
              : Container(
                color: Colors.grey[300],
                child: const Center(child: Icon(Icons.photo_library)),
              ),
    );
  }
}
