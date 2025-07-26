import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_picker/image_picker.dart';

import '../../controller/home_bloc/home_bloc.dart';

enum SelectedTab {
  similar,
  context,
  objects,
  faces,
  color,
  background,
  docs,
  only_people,
}

class SearchResultsScreen extends StatefulWidget {
  SelectedTab selectedTab = SelectedTab.context;
  final Map<String, dynamic> results;

  SearchResultsScreen({required this.results});

  @override
  State<SearchResultsScreen> createState() => _SearchResultsScreenState();
}

class _SearchResultsScreenState extends State<SearchResultsScreen> {
  List<File> _thinkDeeperResults = [];
  bool _isThinkDeeperLoading = false;

  void _loadThinkDeeperResults() async {
    setState(() => _isThinkDeeperLoading = true);

    try {
      final similarImagesDir = Directory('similar_images');
      if (await similarImagesDir.exists()) {
        final files =
            await similarImagesDir
                .list()
                .where(
                  (entity) =>
                      entity.path.endsWith('.jpg') ||
                      entity.path.endsWith('.png'),
                )
                .map((entity) => File(entity.path))
                .toList();

        // Sort files by their prefix number (0001_, 0002_, etc.)
        files.sort((a, b) {
          final aNum =
              int.tryParse(a.path.split('/').last.split('_').first) ?? 0;
          final bNum =
              int.tryParse(b.path.split('/').last.split('_').first) ?? 0;
          return aNum.compareTo(bNum);
        });

        setState(() => _thinkDeeperResults = files);
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error loading results: ${e.toString()}')),
      );
    } finally {
      setState(() => _isThinkDeeperLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Search Results')),
      floatingActionButton:
          widget.selectedTab == SelectedTab.context
              ? BlocConsumer<HomeBloc, HomeState>(
                listener: (context, state) {
                  if (state is ThinkDeeperFailure) {
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(SnackBar(content: Text(state.error)));
                  }
                },
                builder: (context, state) {
                  if (state is ThinkDeeperLoading) {
                    return Padding(
                      padding: const EdgeInsets.only(right: 30.0),
                      child: FloatingActionButton(
                        onPressed: () {},
                        child: const Padding(
                          padding: EdgeInsets.all(12),
                          child: Center(child: CircularProgressIndicator()),
                        ),
                        tooltip: 'loading now\nits going to take about 15 min',
                      ),
                    );
                  } else {
                    return SizedBox(
                      width: 150,
                      child: FloatingActionButton(
                        onPressed: () {
                          context.read<HomeBloc>().add(
                            ThinkDeeper(
                              searchText: searchText,
                              imageFile: imageFile,
                            ),
                          );
                          _loadThinkDeeperResults();
                        },
                        child: Text("Think Deeper"),
                        tooltip: 'its going to take about 15 min',
                      ),
                    );
                  }
                },
              )
              : null,
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Left Column - Search Controls
          Container(
            width: 250,
            padding: EdgeInsets.all(16),

            child: Column(
              children: [
                // Search Box
                Hero(
                  tag: 'search-box',
                  child: Material(
                    type: MaterialType.transparency,
                    child: TextField(
                      decoration: InputDecoration(
                        labelText: 'Search Text',
                        border: OutlineInputBorder(),
                        filled: true,
                      ),
                      controller: TextEditingController(text: searchText),
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // Image Preview
                BlocConsumer<HomeBloc, HomeState>(
                  listener: (context, state) {
                    if (state is PickImageFailure) {
                      ScaffoldMessenger.of(
                        context,
                      ).showSnackBar(SnackBar(content: Text(state.error)));
                    }
                  },
                  builder: (context, state) {
                    if (state is PickImageLoading) {
                      return const Padding(
                        padding: EdgeInsets.all(12),
                        child: Center(child: CircularProgressIndicator()),
                      );
                    } else if (imageFile == null) {
                      return Text('');
                    }
                    return Hero(
                      tag: 'search-image',
                      child: Material(
                        type: MaterialType.transparency,
                        child: Column(
                          children: [
                            Container(
                              height: 150,
                              width: double.infinity,
                              decoration: BoxDecoration(
                                border: Border.all(color: Colors.grey),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(8),
                                child: Image.file(
                                  imageFile!,
                                  fit: BoxFit.cover,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
                const SizedBox(height: 10),
                OutlinedButton.icon(
                  icon: const Icon(Icons.image_search),
                  label: const Text('Chose Image'),
                  onPressed: () {
                    context.read<HomeBloc>().add(PickImage());
                  },
                ),
                const SizedBox(height: 20),
                BlocConsumer<HomeBloc, HomeState>(
                  listener: (context, state) {
                    if (state is SearchFailure) {
                      ScaffoldMessenger.of(
                        context,
                      ).showSnackBar(SnackBar(content: Text(state.error)));
                    }
                  },
                  builder: (context, state) {
                    if (state is SearchLoading) {
                      return const Padding(
                        padding: EdgeInsets.all(12),
                        child: Center(child: CircularProgressIndicator()),
                      );
                    } else {
                      // Search Button
                      return ElevatedButton(
                        onPressed: () {
                          // Trigger search again
                          context.read<HomeBloc>().add(
                            PerformSearch(
                              searchText: searchText,
                              imageFile: imageFile,
                            ),
                          );
                        },
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(double.infinity, 50),
                        ),
                        child: const Text('Search Again'),
                      );
                    }
                  },
                ),
              ],
            ),
          ),

          // Middle Column - Filters and Results
          Expanded(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Filter Options
                  const Text(
                    'Filter By:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 8,
                    children: [
                      FilterChip(
                        label: const Text('Similar'),
                        selected: widget.selectedTab == SelectedTab.similar,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.similar;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Context'),
                        selected: widget.selectedTab == SelectedTab.context,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.context;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Objects'),
                        selected: widget.selectedTab == SelectedTab.objects,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.objects;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Faces'),
                        selected: widget.selectedTab == SelectedTab.faces,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.faces;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Color'),
                        selected: widget.selectedTab == SelectedTab.color,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.color;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Background'),
                        selected: widget.selectedTab == SelectedTab.background,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.background;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Documents | Books | Text'),
                        selected: widget.selectedTab == SelectedTab.docs,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.docs;
                          });
                        },
                      ),
                      FilterChip(
                        label: const Text('Only People'),
                        selected: widget.selectedTab == SelectedTab.only_people,
                        onSelected: (bool value) {
                          setState(() {
                            widget.selectedTab = SelectedTab.only_people;
                          });
                        },
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),

                  // Results Grid
                  Expanded(
                    child:
                        _isThinkDeeperLoading
                            ? Center(child: CircularProgressIndicator())
                            : _buildResultsGrid(),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsGrid() {
    // Show think deeper results if in context tab and results exist
    if (widget.selectedTab == SelectedTab.context &&
        _thinkDeeperResults.isNotEmpty) {
      return GridView.builder(
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
          crossAxisCount: 3,
          crossAxisSpacing: 8,
          mainAxisSpacing: 8,
          childAspectRatio: 1,
        ),
        itemCount: _thinkDeeperResults.length,
        itemBuilder: (context, index) {
          return Card(
            elevation: 2,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Expanded(
                  child: Image.file(
                    _thinkDeeperResults[index],
                    fit: BoxFit.cover,
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(
                    'Rank: ${index + 1}',
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
          );
        },
      );
    }

    // Default results grid for other tabs
    return GridView.builder(
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        crossAxisSpacing: 8,
        mainAxisSpacing: 8,
        childAspectRatio: 1,
      ),
      itemCount: widget.results["output"][widget.selectedTab.name].length,
      itemBuilder: (context, index) {
        return Card(
          elevation: 2,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: Image.file(
                  File(
                    widget.results['output'][widget.selectedTab.name][index],
                  ),
                  fit: BoxFit.cover,
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text(
                  'Similarity: ${widget.results['output'][widget.selectedTab.name][index]}',
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}
