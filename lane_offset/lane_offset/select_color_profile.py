"""Interactive command for selecting the persistent lane color profile."""

import yaml

from .color_profiles import read_profiles, select_profile, selected_profile_name


def main():
    try:
        data, path = read_profiles()
        current = selected_profile_name(data)
        print(f'\n색상 프로필 파일: {path}')
        print(f'현재 선택: {current}\n')
        print('저장된 장소:')
        names = list(data['profiles'])
        for index, name in enumerate(names, 1):
            description = data['profiles'][name].get('description', '')
            marker = ' (현재)' if name == current else ''
            print(f'  {index}. {name}{marker}  {description}')

        answer = input('\n장소 이름 또는 번호를 입력하세요 (취소: Enter): ').strip()
        if not answer:
            print('변경하지 않았습니다.')
            return
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            answer = names[int(answer) - 1]
        select_profile(answer, path)
        print(f"'{answer}' 프로필을 선택했습니다. 차선 노드를 재시작하면 적용됩니다.")
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f'색상 프로필을 변경하지 못했습니다: {error}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
