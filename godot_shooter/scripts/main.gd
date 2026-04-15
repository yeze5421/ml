extends Node2D

enum GameState { MENU, PLAYING, PAUSED, GAME_OVER }

const PLAYER_SCENE := preload("res://scenes/Player.tscn")
const ENEMY_SCENE := preload("res://scenes/Enemy.tscn")
const BOSS_SCENE := preload("res://scenes/Boss.tscn")
const BULLET_SCENE := preload("res://scenes/Bullet.tscn")
const EXPLOSION_SCENE := preload("res://scenes/Explosion.tscn")

@onready var gameplay: Node2D = $Gameplay
@onready var player_holder: Node2D = $Gameplay/PlayerHolder
@onready var enemy_holder: Node2D = $Gameplay/Enemies
@onready var bullet_holder: Node2D = $Gameplay/Bullets
@onready var fx_holder: Node2D = $Gameplay/Effects

@onready var enemy_timer: Timer = $EnemySpawnTimer
@onready var wave_timer: Timer = $WaveTimer

@onready var menu_layer: Control = $UI/MainMenu
@onready var hud: Control = $UI/HUD
@onready var pause_layer: Control = $UI/PauseMenu
@onready var game_over_layer: Control = $UI/GameOver

@onready var health_label: Label = $UI/HUD/MarginContainer/VBoxContainer/HealthLabel
@onready var score_label: Label = $UI/HUD/MarginContainer/VBoxContainer/ScoreLabel
@onready var boss_hp_label: Label = $UI/HUD/MarginContainer/VBoxContainer/BossLabel
@onready var final_score_label: Label = $UI/GameOver/Panel/VBoxContainer/FinalScoreLabel

var state := GameState.MENU
var score := 0
var difficulty := 1.0
var player: Player
var boss: Boss
var boss_spawned := false

func _ready() -> void:
	randomize()
	_connect_ui()
	_show_menu()

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("pause") and state in [GameState.PLAYING, GameState.PAUSED]:
		if state == GameState.PLAYING:
			_pause_game()
		else:
			_resume_game()

func _process(_delta: float) -> void:
	if state == GameState.PLAYING and not boss_spawned and score >= 3000:
		spawn_boss()

func _connect_ui() -> void:
	$UI/MainMenu/Panel/VBoxContainer/StartButton.pressed.connect(_on_start_pressed)
	$UI/MainMenu/Panel/VBoxContainer/QuitButton.pressed.connect(func(): get_tree().quit())
	$UI/PauseMenu/Panel/VBoxContainer/ResumeButton.pressed.connect(_resume_game)
	$UI/PauseMenu/Panel/VBoxContainer/RestartButton.pressed.connect(start_game)
	$UI/PauseMenu/Panel/VBoxContainer/MenuButton.pressed.connect(_show_menu)
	$UI/GameOver/Panel/VBoxContainer/RestartButton.pressed.connect(start_game)
	$UI/GameOver/Panel/VBoxContainer/MenuButton.pressed.connect(_show_menu)
	enemy_timer.timeout.connect(_on_enemy_spawn_timer)
	wave_timer.timeout.connect(_on_wave_timer)

func _on_start_pressed() -> void:
	start_game()

func start_game() -> void:
	clear_gameplay()
	score = 0
	difficulty = 1.0
	boss_spawned = false
	boss = null
	state = GameState.PLAYING
	menu_layer.visible = false
	pause_layer.visible = false
	game_over_layer.visible = false
	hud.visible = true
	boss_hp_label.visible = false
	Engine.time_scale = 1.0

	player = PLAYER_SCENE.instantiate()
	player.global_position = Vector2(480, 460)
	player.setup_play_rect(Rect2(Vector2.ZERO, Vector2(960, 540)))
	player.health_changed.connect(_update_health)
	player.request_shot.connect(_spawn_player_bullet)
	player.died.connect(_on_player_died)
	player_holder.add_child(player)

	enemy_timer.wait_time = 1.2
	enemy_timer.start()
	wave_timer.start()
	_update_health(player.max_health)
	_update_score(0)

func _show_menu() -> void:
	state = GameState.MENU
	Engine.time_scale = 1.0
	clear_gameplay()
	menu_layer.visible = true
	hud.visible = false
	pause_layer.visible = false
	game_over_layer.visible = false
	enemy_timer.stop()
	wave_timer.stop()

func _pause_game() -> void:
	state = GameState.PAUSED
	Engine.time_scale = 0.0
	pause_layer.visible = true

func _resume_game() -> void:
	if state != GameState.PAUSED:
		return
	state = GameState.PLAYING
	Engine.time_scale = 1.0
	pause_layer.visible = false

func _game_over() -> void:
	state = GameState.GAME_OVER
	Engine.time_scale = 1.0
	enemy_timer.stop()
	wave_timer.stop()
	pause_layer.visible = false
	hud.visible = false
	game_over_layer.visible = true
	final_score_label.text = "最终分数: %d" % score

func _on_player_died() -> void:
	spawn_explosion(player.global_position)
	_game_over()

func _on_enemy_spawn_timer() -> void:
	if state != GameState.PLAYING:
		return
	var enemy_count := randi_range(1, 2 + int(difficulty))
	for i in enemy_count:
		var enemy := ENEMY_SCENE.instantiate() as Enemy
		enemy.global_position = Vector2(randf_range(40, 920), randf_range(-120, -40))
		enemy.speed += difficulty * 18.0
		enemy.health += int(difficulty * 6.0)
		enemy.points += int(difficulty * 20.0)
		enemy.died.connect(_on_enemy_died)
		enemy.request_shot.connect(_spawn_enemy_bullet)
		enemy.area_entered.connect(_on_enemy_area_entered.bind(enemy))
		enemy_holder.add_child(enemy)

func _on_wave_timer() -> void:
	if state != GameState.PLAYING:
		return
	difficulty += 0.25
	enemy_timer.wait_time = maxf(0.35, enemy_timer.wait_time - 0.05)

func _spawn_player_bullet(pos: Vector2) -> void:
	var bullet := BULLET_SCENE.instantiate() as Bullet
	bullet.from_enemy = false
	bullet.direction = Vector2.UP
	bullet.global_position = pos
	bullet.area_entered.connect(_on_player_bullet_hit.bind(bullet))
	bullet_holder.add_child(bullet)

func _spawn_enemy_bullet(pos: Vector2, dir := Vector2.DOWN) -> void:
	var bullet := BULLET_SCENE.instantiate() as Bullet
	bullet.from_enemy = true
	bullet.direction = dir
	bullet.speed = 360
	bullet.damage = 12
	bullet.global_position = pos
	bullet.area_entered.connect(_on_enemy_bullet_hit.bind(bullet))
	bullet_holder.add_child(bullet)

func _on_player_bullet_hit(area: Area2D, bullet: Bullet) -> void:
	if not is_instance_valid(bullet):
		return
	if area is Enemy:
		(area as Enemy).take_damage(bullet.damage)
		bullet.queue_free()
	elif area is Boss:
		(area as Boss).take_damage(bullet.damage)
		bullet.queue_free()

func _on_enemy_bullet_hit(area: Area2D, bullet: Bullet) -> void:
	if not is_instance_valid(bullet):
		return
	if area is Player:
		(area as Player).take_damage(bullet.damage)
		spawn_explosion(bullet.global_position, 0.6)
		bullet.queue_free()

func _on_enemy_area_entered(area: Area2D, enemy: Enemy) -> void:
	if area is Player:
		(area as Player).take_damage(20)
		if is_instance_valid(enemy):
			spawn_explosion(enemy.global_position)
			enemy.queue_free()

func _on_enemy_died(points: int, pos: Vector2) -> void:
	_update_score(score + points)
	spawn_explosion(pos)

func spawn_boss() -> void:
	boss_spawned = true
	boss = BOSS_SCENE.instantiate() as Boss
	boss.global_position = Vector2(480, 90)
	boss.health_changed.connect(_on_boss_health_changed)
	boss.request_shot.connect(_spawn_enemy_bullet)
	boss.died.connect(_on_boss_died)
	boss.area_entered.connect(_on_boss_area_entered)
	enemy_holder.add_child(boss)
	boss_hp_label.visible = true
	_on_boss_health_changed(boss.max_health)

func _on_boss_area_entered(area: Area2D) -> void:
	if area is Player:
		(area as Player).take_damage(35)

func _on_boss_health_changed(value: int) -> void:
	boss_hp_label.text = "Boss HP: %d" % value

func _on_boss_died(points: int, pos: Vector2) -> void:
	_update_score(score + points)
	spawn_explosion(pos, 1.8)
	boss_hp_label.visible = false
	await get_tree().create_timer(0.8).timeout
	_game_over()

func _update_health(value: int) -> void:
	health_label.text = "生命值: %d" % value

func _update_score(new_score: int) -> void:
	score = new_score
	score_label.text = "分数: %d" % score

func spawn_explosion(pos: Vector2, scale_mul := 1.0) -> void:
	var fx := EXPLOSION_SCENE.instantiate()
	fx.global_position = pos
	fx.scale = Vector2.ONE * scale_mul
	fx_holder.add_child(fx)

func clear_gameplay() -> void:
	for n in [player_holder, enemy_holder, bullet_holder, fx_holder]:
		for c in n.get_children():
			c.queue_free()
