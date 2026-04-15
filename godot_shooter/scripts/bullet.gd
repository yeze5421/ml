extends Area2D
class_name Bullet

@export var speed: float = 650.0
@export var damage: int = 10
@export var direction := Vector2.UP
@export var from_enemy := false

func _ready() -> void:
	if from_enemy:
		collision_layer = 8
		collision_mask = 1
	else:
		collision_layer = 2
		collision_mask = 4

func _process(delta: float) -> void:
	global_position += direction.normalized() * speed * delta
	if global_position.y < -20 or global_position.y > 560 or global_position.x < -20 or global_position.x > 980:
		queue_free()
