extends Area2D
class_name Boss

signal health_changed(value: int)
signal died(points: int, pos: Vector2)
signal request_shot(pos: Vector2, dir: Vector2)

@export var max_health: int = 400
@export var points: int = 2000
@export var speed: float = 100.0
@export var shoot_interval: float = 0.55

var health: int
var _dir := 1.0
var _shoot_timer := 0.0

func _ready() -> void:
	health = max_health
	health_changed.emit(health)
	_shoot_timer = shoot_interval

func _process(delta: float) -> void:
	global_position.x += _dir * speed * delta
	if global_position.x < 120:
		_dir = 1.0
	elif global_position.x > 840:
		_dir = -1.0

	_shoot_timer -= delta
	if _shoot_timer <= 0.0:
		_shoot_timer = shoot_interval
		request_shot.emit(global_position + Vector2(0, 32), Vector2.DOWN)
		request_shot.emit(global_position + Vector2(-26, 30), Vector2(-0.25, 1.0))
		request_shot.emit(global_position + Vector2(26, 30), Vector2(0.25, 1.0))

func take_damage(amount: int) -> void:
	health = max(0, health - amount)
	health_changed.emit(health)
	if health <= 0:
		died.emit(points, global_position)
		queue_free()
