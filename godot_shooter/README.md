# Sky Blaster 2D (Godot 4)

一个可直接运行的 Godot 4 2D 打飞机桌面小游戏示例，使用占位美术资源，包含完整可扩展的项目结构。

## 已实现功能

- 开始菜单（开始/退出）
- 玩家 WASD 移动、空格连续射击
- 敌机持续刷怪（随时间逐步提升难度）
- 玩家血量、分数 HUD
- 爆炸特效（CPUParticles2D）
- Boss 战（分数达到阈值后出现）
- 暂停（P 键 + 暂停菜单）
- 游戏结束界面 + 重新开始
- Windows 导出预设（`export_presets.cfg`）

## 目录结构

```text
godot_shooter/
├─ assets/
│  └─ icon.svg
├─ scenes/
│  ├─ Main.tscn
│  ├─ Player.tscn
│  ├─ Enemy.tscn
│  ├─ Bullet.tscn
│  ├─ Boss.tscn
│  └─ Explosion.tscn
├─ scripts/
│  ├─ main.gd
│  ├─ player.gd
│  ├─ enemy.gd
│  ├─ bullet.gd
│  ├─ boss.gd
│  └─ explosion.gd
├─ export_presets.cfg
└─ project.godot
```

## 运行

1. 使用 Godot 4.2+ 打开 `godot_shooter/project.godot`
2. 点击运行（F5）

## Windows 导出

1. 安装 Godot Windows Export Templates（Editor -> Manage Export Templates）
2. 打开 `Project -> Export...`
3. 选择 `Windows Desktop` 预设
4. 导出路径已默认配置为：`build/SkyBlaster.exe`

> 若首次导出失败，通常是因为未安装对应版本模板或版本不匹配。
