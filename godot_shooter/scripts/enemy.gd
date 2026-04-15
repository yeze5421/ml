extends Area2D
class_name Enemy

signal died(points: int, pos: Vector2)
signal request_shot(pos: Vector2)

@export var speed: float = 120.0
@export var health: int = 25
@export var points: int = 100
@export var shoot_interval: float = 1.8

var _shoot_timer := 0.0

func _ready() -> void:
	_shoot_timer = randf_range(0.3, shoot_interval)

func _process(delta: float) -> void:
	global_position.y += speed * delta
	_shoot_timer -= delta
	if _shoot_timer <= 0.0:
		_shoot_timer = shoot_interval
		request_shot.emit(global_position + Vector2(0, 20))

	if global_position.y > 580:
		queue_free()

func take_damage(amount: int) -> void:
	health -= amount
	if health <= 0:
		died.emit(points, global_position)
		queue_free()
