import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:image_search_engine_front_end/presentation/screen/albums_screen.dart';
import 'package:image_search_engine_front_end/presentation/screen/editor_screen.dart';
import 'package:image_search_engine_front_end/presentation/screen/home_screen.dart';

import '../../data/theme/theme_bloc.dart';

class TemplateScreen extends StatefulWidget {
  @override
  _TemplateScreenState createState() => _TemplateScreenState();
}

class _TemplateScreenState extends State<TemplateScreen> {
  bool _isCollapsed = false;
  int _selectedIndex = 0;

  final List<Widget> _pages = [HomeScreen(), AlbumScreen(), EditorScreen()];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Row(
          children: [
            // Sidebar
            CollapsibleSidebar(
              isCollapsed: _isCollapsed,
              selectedIndex: _selectedIndex,
              onItemSelected: (index) {
                setState(() {
                  _selectedIndex = index;
                });
              },
              onCollapseToggled: () {
                setState(() {
                  _isCollapsed = !_isCollapsed;
                });
              },
            ),

            // Main content
            Expanded(child: _pages[_selectedIndex]),
          ],
        ),
      ),
    );
  }
}

class CollapsibleSidebar extends StatelessWidget {
  final bool isCollapsed;
  final int selectedIndex;
  final Function(int) onItemSelected;
  final VoidCallback onCollapseToggled;

  const CollapsibleSidebar({
    required this.isCollapsed,
    required this.selectedIndex,
    required this.onItemSelected,
    required this.onCollapseToggled,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: isCollapsed ? 80 : 250,
      color: Theme.of(context).secondaryHeaderColor,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header with collapse button
          SidebarHeader(
            isCollapsed: isCollapsed,
            onCollapseToggled: onCollapseToggled,
          ),
          Divider(color: Theme.of(context).secondaryHeaderColor, height: 1),
          // Navigation items
          Expanded(
            child: SidebarMenu(
              isCollapsed: isCollapsed,
              selectedIndex: selectedIndex,
              onItemSelected: onItemSelected,
            ),
          ),
          Padding(
            padding: EdgeInsets.all(10),
            child: Switch(
              value: context.read<ThemeBloc>().state == ThemeMode.dark,
              onChanged: (value) {
                context.read<ThemeBloc>().add(ThemeChanged(value));
              },
            ),
          ),
        ],
      ),
    );
  }
}

class SidebarHeader extends StatelessWidget {
  final bool isCollapsed;
  final VoidCallback onCollapseToggled;

  const SidebarHeader({
    required this.isCollapsed,
    required this.onCollapseToggled,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(vertical: 20, horizontal: 10),
      child:
          isCollapsed
              ? IconButton(icon: Icon(Icons.menu), onPressed: onCollapseToggled)
              : Row(
                children: [
                  Text(
                    'VisionAI',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  Spacer(),
                  IconButton(
                    icon: Icon(Icons.menu),
                    onPressed: onCollapseToggled,
                  ),
                ],
              ),
    );
  }
}

class SidebarMenu extends StatelessWidget {
  final bool isCollapsed;
  final int selectedIndex;
  final Function(int) onItemSelected;

  const SidebarMenu({
    required this.isCollapsed,
    required this.selectedIndex,
    required this.onItemSelected,
  });

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: EdgeInsets.symmetric(vertical: 10),
      children: [
        SidebarItem(
          isCollapsed: isCollapsed,
          icon: Icons.home,
          title: 'Home',
          isSelected: selectedIndex == 0,
          onTap: () => onItemSelected(0),
        ),
        SidebarItem(
          isCollapsed: isCollapsed,
          icon: Icons.photo_album,
          title: 'Albums',
          isSelected: selectedIndex == 1,
          onTap: () => onItemSelected(1),
        ),
        SidebarItem(
          isCollapsed: isCollapsed,
          icon: Icons.edit,
          title: 'Image Editor',
          isSelected: selectedIndex == 2,
          onTap: () => onItemSelected(2),
        ),
      ],
    );
  }
}

class SidebarItem extends StatelessWidget {
  final bool isCollapsed;
  final IconData icon;
  final String title;
  final bool isSelected;
  final VoidCallback onTap;

  const SidebarItem({
    required this.isCollapsed,
    required this.icon,
    required this.title,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        margin: EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        padding: EdgeInsets.symmetric(horizontal: 10, vertical: 15),
        decoration: BoxDecoration(
          color:
              isSelected
                  ? Theme.of(context).primaryColor.withAlpha(50)
                  : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child:
            isCollapsed
                ? Icon(icon)
                : Row(
                  children: [
                    Icon(icon),
                    SizedBox(width: 15),
                    Text(title, style: TextStyle(fontSize: 16)),
                  ],
                ),
      ),
    );
  }
}
